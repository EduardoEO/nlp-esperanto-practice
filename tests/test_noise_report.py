"""Pruebas unitarias para la herramienta de auditoría de ruido (noise_report.py).

Evalúa la generación determinista de muestras estratificadas, la sesión interactiva
de anotación manual (filtros, saltos, reanudación e interrupciones), la protección
contra sobrescritura accidental y el cálculo de métricas sobre muestras anotadas.
"""
import pytest
import noise_report as nr


def sample(tmp_path, labels=('', '', ''), classes=('fiziko', 'fiziko', 'kemio')):
    """Crea un archivo CSV de muestra temporal con las etiquetas y clases indicadas."""
    path = tmp_path / 'sample.csv'
    nr.save_sample(path, ['id', 'class', 'text', 'generic'], [
        dict(id=str(i), **{'class': c}, text='La ĉelo; teksto kun ĝ.', generic=v)
        for i, (v, c) in enumerate(zip(labels, classes))])
    return path


def answers(monkeypatch, values):
    """Simula respuestas secuenciales del usuario por consola reemplazando builtins.input."""
    it = iter(values)
    monkeypatch.setattr('builtins.input', lambda _: next(it))


def test_annotation_resume_and_filter(tmp_path, monkeypatch):
    """Verifica el filtrado por clase, la salida con 'q' y la reanudación incremental."""
    p = sample(tmp_path)
    answers(monkeypatch, ['1', 'q'])
    nr.annotate(p, 'fiziko')
    assert [r['generic'] for r in nr.read_sample(p)[1]] == ['1', '', '']
    answers(monkeypatch, ['2'])
    nr.annotate(p, 'fiziko')
    assert [r['generic'] for r in nr.read_sample(p)[1]] == ['1', '2', '']
    assert nr.read_sample(p)[1][0]['text'] == 'La ĉelo; teksto kun ĝ.'


def test_skip_and_invalid_answer(tmp_path, monkeypatch):
    """Verifica el salto de filas ('s') y la repetición de pregunta tras entrada inválida."""
    p = sample(tmp_path)
    answers(monkeypatch, ['s', 'bad', '0'])
    nr.annotate(p, 'fiziko')
    assert [r['generic'] for r in nr.read_sample(p)[1]] == ['', '0', '']


def test_eof_preserves_saved_answers(tmp_path, monkeypatch):
    """Verifica que una interrupción abrupta (EOFError) conserve las respuestas ya guardadas."""
    p = sample(tmp_path)
    def reply(_):
        if nr.read_sample(p)[1][0]['generic'] == '':
            return '0'
        raise EOFError
    monkeypatch.setattr('builtins.input', reply)
    nr.annotate(p)
    assert nr.read_sample(p)[1][0]['generic'] == '0'


def test_incomplete_analysis(tmp_path, capsys):
    """Verifica el cálculo de ruido parcial y el aviso 'INCOMPLETO' si quedan filas pendientes."""
    nr.analyze_sample(sample(tmp_path, ('0', '2', '')))
    output = capsys.readouterr().out
    assert 'ruido=50.0%' in output
    assert 'INCOMPLETO' in output
    assert 'Sin anotaciones' in output


def test_complete_analysis(tmp_path, capsys):
    """Verifica el cálculo de ruido total (genérico + otra temática) en muestras completas."""
    nr.analyze_sample(sample(tmp_path, ('0', '1', '2')))
    output = capsys.readouterr().out
    assert 'ruido=66.7%' in output
    assert 'INCOMPLETO' not in output


def test_existing_sample_is_preserved(tmp_path):
    """Verifica que --sample aborte sin sobrescribir si el archivo de muestra ya existe."""
    p = sample(tmp_path, ('0', '', ''))
    before = p.read_bytes()
    with pytest.raises(SystemExit):
        nr.main(['--sample', '50', '--out', str(p)])
    assert p.read_bytes() == before


def test_unknown_class_does_not_modify(tmp_path):
    """Verifica que filtrar por una clase inexistente lance ValueError sin alterar el archivo."""
    p = sample(tmp_path)
    before = p.read_bytes()
    with pytest.raises(ValueError, match='Clase desconocida'):
        nr.annotate(p, 'unknown')
    assert p.read_bytes() == before


def test_invalid_label_is_rejected(tmp_path):
    """Verifica que read_sample rechace valores fuera del conjunto permitido {'', '0', '1', '2'}."""
    with pytest.raises(ValueError, match='Etiqueta inválida'):
        nr.read_sample(sample(tmp_path, ('9', '', '')))


def test_analysis_does_not_need_dataset(tmp_path, capsys):
    """Verifica que el modo --analyze funcione leyendo solo la muestra sin requerir dataset.csv."""
    p = sample(tmp_path)
    assert nr.main(['--analyze', '--out', str(p), '--data', 'missing.csv']) == 0
    assert 'Sin anotaciones' in capsys.readouterr().out


def test_sample_generation_is_deterministic(tmp_path):
    """Verifica que la extracción de muestras con --sample sea determinista dada la semilla."""
    data = tmp_path / 'data.csv'
    data.write_text('text;class\nLa ĉelo vivas.;biologio\nLa birdo flugas.;biologio\nLa forto agas.;fiziko\n', encoding='utf-8')
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    for out in (a, b):
        assert nr.main(['--sample', '1', '--data', str(data), '--out', str(out)]) == 0
    assert a.read_bytes() == b.read_bytes()
    assert len(nr.read_sample(a)[1]) == 2
