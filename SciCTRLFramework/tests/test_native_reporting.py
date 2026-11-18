"""Reporting checks do not invoke models or recreate manuscript fragments."""
import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def reporting():
    script = Path(__file__).resolve().parents[2] / "Method" / "summarize_native_runs.py"
    spec = importlib.util.spec_from_file_location('native_reporting_test_module', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_preserves_single_file_and_surroundings(reporting, tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, 'PAPER', tmp_path)
    main = tmp_path / 'iclr2027_conference.tex'
    main.write_text('before\n% BEGIN INLINE: audit.tex\nold\n% END INLINE: audit.tex\nafter\n')
    reporting.update_inline('audit.tex', 'new\n')
    assert main.read_text() == 'before\n% BEGIN INLINE: audit.tex\nnew\n% END INLINE: audit.tex\nafter\n'
    assert list(tmp_path.glob('*.tex')) == [main]


def test_missing_markers_do_not_overwrite(reporting, tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, 'PAPER', tmp_path)
    main = tmp_path / 'iclr2027_conference.tex'
    main.write_text('preserve me')
    with pytest.raises(ValueError):
        reporting.update_inline('audit.tex', 'replacement')
    assert main.read_text() == 'preserve me'


def test_artifact_dispositions_are_distinct(reporting):
    classify = reporting.artifact_category
    assert classify({'numerical_valid': True}) == 'numerical_pass'
    assert classify({'heldout': {'valid': False}}) == 'numerical_mismatch'
    assert classify({'artifact_error': 'required final analysis.py absent'}) == 'missing_artifact'
    assert classify({'artifact_error': 'ValueError: call unavailable'}) == 'execution_or_contract_rejection'
    assert classify({'error': 'RuntimeError'}) == 'evaluator_error'
    assert classify({}) == 'unresolved'
