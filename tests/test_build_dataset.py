"""Pruebas del ETL contra una Wikipedia SIMULADA (sin red)."""
import csv
import json

import build_dataset as bd
from esperanto_nlp.contract import validate_file

PARA = ("La {t} estas tre grava en la sporta mondo kaj multaj spektantoj sekvas la ludojn "
        "ĉiusemajne kun granda intereso kaj ĝojo, kiel montras la historio de la sporto {n}.")


def make_article(title, n_paragraphs=4):
    body = "\n".join(PARA.format(t=title, n=i) for i in range(n_paragraphs))
    return ("Mallonga enkonduko.\n" + body + "\n\n== Historio ==\n" + body.split("\n")[0] +
            " Plia teksto ĉi tie ĝis la fino.\n\n== Referencoj ==\n" + PARA.format(t="REF", n=99) +
            "\n=== Subsekcio ===\n" + PARA.format(t="SUB", n=98))


# categoría -> (artículos, subcategorías)
WIKI = {
    "Alfa": ([1, 2], ["Beta"]), "Beta": ([3], []),
    "Gamo": ([4, 5, 2], []),               # el artículo 2 está también en "Alfa" -> ambiguo
}
TITLES = {1: "Unua", 2: "Dua", 3: "Tria", 4: "Kvara", 5: "Kvina"}


class FakeResponse:
    def __init__(self, data):
        self._d = data
    def raise_for_status(self):
        pass
    def json(self):
        return self._d


class FakeSession:
    def __init__(self):
        self.calls = 0
    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if params.get("list") == "categorymembers":
            cat = params["cmtitle"].split(":", 1)[1]
            pages, subs = WIKI.get(cat, ([], []))
            if params["cmtype"] == "page":
                members = [{"pageid": p, "ns": 0, "title": TITLES[p]} for p in pages]
                if "cmcontinue" not in params and len(members) > 1:       # simula paginación
                    return FakeResponse({"query": {"categorymembers": members[:1]},
                                         "continue": {"cmcontinue": "x"}})
                if "cmcontinue" in params:
                    members = members[1:]
            else:
                members = [{"pageid": 1000 + i, "ns": 14, "title": f"Kategorio:{s}"} for i, s in enumerate(subs)]
            return FakeResponse({"query": {"categorymembers": members}})
        if params.get("prop") == "categoryinfo":
            pages = []
            for title in params["titles"].split("|"):
                n = title.split(":", 1)[1]
                arts, subs = WIKI.get(n, ([], []))
                page = {"title": title}
                if arts or subs:                      # categoría vacía: MediaWiki omite categoryinfo
                    page["categoryinfo"] = {"pages": len(arts), "subcats": len(subs)}
                pages.append(page)
            return FakeResponse({"query": {"pages": pages}})
        pid = int(params["pageids"])
        return FakeResponse({"query": {"pages": [{
            "pageid": pid, "title": TITLES[pid], "lastrevid": 7000 + pid,
            "extract": make_article(TITLES[pid])}]}})


CFG = {
    "wiki_api": "http://fake", "category_prefix": "Kategorio:", "user_agent": "t", "seed": 42,
    "request_delay_seconds": 0, "min_chars": 100, "max_chars": 1500, "min_words": 12,
    "max_paragraphs_per_article": 3, "target_paragraphs_per_class": 100, "max_category_depth": 2,
    "max_candidates_per_class": 100, "balance_classes": True,
    "classes": {"alfa": ["Alfa"], "gamo": ["Gamo"]},
}


def test_extract_paragraphs_skips_headings_and_reference_sections():
    paras = bd.extract_paragraphs(make_article("X"))
    joined = " ".join(paras)
    assert "REF" not in joined and "SUB" not in joined and "==" not in joined
    assert len(paras) == 6        # intro + 4 + 1 de "Historio"


def test_article_paragraphs_limit_and_determinism():
    art = {"pageid": 5, "title": "T", "extract": make_article("Test", 8)}
    a = bd.article_paragraphs(art, CFG)
    b = bd.article_paragraphs(art, CFG)
    assert a == b and len(a) == 3
    assert all(";" not in p and len(p) >= 100 for p in a)


def test_crawl_follows_subcategories_and_pagination():
    pages = bd.crawl_category(FakeSession(), CFG, "Alfa")
    assert pages == {1: "Unua", 2: "Dua", 3: "Tria"}
    assert bd.crawl_category(FakeSession(), {**CFG, "max_category_depth": 0}, "Alfa") == {1: "Unua", 2: "Dua"}


def test_drop_ambiguous():
    out = bd.drop_ambiguous({"a": {1: "x", 2: "y"}, "g": {2: "y", 4: "z"}})
    assert out == {"a": {1: "x"}, "g": {4: "z"}}


def test_full_build_is_valid_deterministic_and_cached(tmp_path):
    s1 = FakeSession()
    rows1 = bd.build_rows(s1, CFG, tmp_path / "raw")
    assert {r["class"] for r in rows1} == {"alfa", "gamo"}
    assert 2 not in {r["pageid"] for r in rows1}          # ambiguo eliminado
    bd.write_outputs(rows1, tmp_path, CFG)
    rep = validate_file(tmp_path / "dataset.csv", min_classes=2, min_per_class=1)
    assert rep.ok, rep.summary()
    with open(tmp_path / "dataset_meta.csv", encoding="utf-8") as f:
        meta = list(csv.DictReader(f, delimiter=";"))
    assert len(meta) == len(rows1) and meta[0]["revid"].startswith("70")
    stats = json.loads((tmp_path / "dataset_stats.json").read_text(encoding="utf-8"))
    assert stats["seed"] == 42 and "user_agent" not in stats["config"]
    # Segunda ejecución: mismo resultado y sin llamadas de artículo (caché)
    s2 = FakeSession()
    rows2 = bd.build_rows(s2, CFG, tmp_path / "raw")
    assert [r["text"] for r in rows1] == [r["text"] for r in rows2]
    assert s2.calls < s1.calls


def test_balance_rows():
    rows = [{"class": "a", "text": str(i)} for i in range(10)] + [{"class": "b", "text": "x"} for _ in range(3)]
    out = bd.balance_rows(rows, ["a", "b"], 1)
    assert sum(r["class"] == "a" for r in out) == 3 and sum(r["class"] == "b" for r in out) == 3


def test_exclude_subcategories_skips_biographies():
    """Las subcategorías listadas en ``exclude_subcategories`` no se recorren."""
    cfg = {**CFG, "exclude_subcategories": ["beta"]}
    found = bd.crawl_category(FakeSession(), cfg, "Alfa")
    assert "Tria" not in found.values()      # estaba dentro de la subcategoría excluida "Beta"
    assert "Unua" in found.values()


def test_main_reports_network_errors_without_traceback(monkeypatch, capsys):
    """Sin red, el script termina con código 2 y un mensaje claro (no un traceback)."""
    def boom(args):
        raise bd.ExtractError("sin red")
    monkeypatch.setattr(bd, "cmd_explore", boom)
    assert bd.main(["explore"]) == 2
    assert "ERROR de red/API" in capsys.readouterr().err


def test_describe_subcategories_lists_sizes_sorted():
    rows = bd.describe_subcategories(FakeSession(), CFG, "Alfa")
    assert rows == [("Beta", 1, 0)]


def test_category_info_handles_empty_category():
    info = bd.category_info(FakeSession(), CFG, ["Alfa", "Vacia"])
    assert info["Alfa"] == {"pages": 2, "subcats": 1}
    assert info["Vacia"] == {"pages": 0, "subcats": 0}


def test_depth_override_replaces_config_value():
    import argparse
    cfg = bd._apply_overrides({"max_category_depth": 2}, argparse.Namespace(depth=0))
    assert cfg["max_category_depth"] == 0
    assert bd._apply_overrides({"max_category_depth": 2}, argparse.Namespace(depth=None))["max_category_depth"] == 2


def test_class_depth_controls_how_deep_each_class_crawls():
    base = {**CFG, "classes": {"x": ["Alfa"]}, "seed": 1}
    shallow = bd.collect_candidates(FakeSession(), {**base, "class_depth": {"x": 0}})
    deep = bd.collect_candidates(FakeSession(), {**base, "class_depth": {"x": 1}})
    assert "Tria" not in shallow["x"].values()          # Tria vive en la subcategoría Beta
    assert "Tria" in deep["x"].values()


def test_depth_flag_overrides_class_depth():
    import argparse
    cfg = bd._apply_overrides({"max_category_depth": 2, "class_depth": {"a": 1}}, argparse.Namespace(depth=0))
    assert cfg["max_category_depth"] == 0 and cfg["class_depth"] == {}
