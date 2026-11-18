# Gramática implementada

| Terminación | Categoría (POS) | Lema | Ejemplo |
|---|---|---|---|
| `-o` (+`j`, `n`) | NOUN | raíz + `o` | hundojn → hundo |
| `-a` (+`j`, `n`) | ADJ | raíz + `a` | grandaj → granda |
| `-e` (+`n`) | ADV | raíz + `e` | rapide → rapide |
| `-i` | VERB (infinitivo) | raíz + `i` | kuri |
| `-as / -is / -os` | VERB (presente / pasado / futuro) | raíz + `i` | kuris → kuri |
| `-us / -u` | VERB (condicional / imperativo) | raíz + `i` | kurus → kuri |
| `-ant- -int- -ont-` + o/a/e | participio activo | **infinitivo** | kuranta → kuri |
| `-at- -it- -ot-` + o/a/e | participio pasivo | **infinitivo** | skribata → skribi |

Formas de *esti* → `AUX`. Numerales compuestos (`dudek`) → `NUM`; ordinales (`dudeka`) → `ADJ`.

**Palabras cerradas (léxico):** `la` (DET), preposiciones (ADP), conjunciones (CCONJ/SCONJ),
partículas (`ne`, `ĉu`), pronombres personales y posesivos, y los 45 **correlativos**
(prefijos `ki- ti- i- ĉi- neni-` × sufijos `-u -o -a -e -am -el -al -es -om`), todos con sus
variantes en `-n` y `-j`.

**Trampas cubiertas por tests:** `kio` (PRON, no NOUN), `tre`/`ĉe` (no regla `-e`),
`kato`/`dato`/`ŝtato` (no son participios), `elefanto`, `kandidato` (excepciones), `mi` vs
infinitivos, `cx → ĉ`, `lux` (no es `lŭ`).

## Afijos derivativos reconocidos (Opción B)
Ejemplos comprobados con el código (`analyze(palabra).base` y `.affixes`):

| Afijo | Significado | Ejemplo | `root_` | `affixes` |
|---|---|---|---|---|
| `mal-` | contrario | malbona | bon | `('mal-',)` |
| `-ej-` | lugar | lernejo | lern | `('-ej',)` |
| `-ist-` (+`-in-`) | profesión | instruistino | instru | `('-ist', '-in')` |
| `-ig-` / `-iĝ-` | hacer / volverse | pliigi, ruĝiĝi | pli, ruĝ | `('-ig',)`, `('-iĝ',)` |
| `-ebl-` / `-ind-` | posible / digno | legebla, ŝatinda | leg, ŝat | `('-ebl',)`, `('-ind',)` |
| `-aĵ-` / `-ism-` / `-eg-` | cosa concreta / doctrina / aumentativo | manĝaĵo, kapitalismo, grandega | manĝ, kapital, grand | |

Regla de seguridad: la raíz resultante debe tener **al menos 3 letras y una vocal**; si no,
no se separa (así `reg-` o `lig-` no se rompen). `-ec-` solo se detecta dentro de una palabra
completa (`beleco → bel`).
