import math
import pytest
from run_validation_experiments import blind_input, checked_prediction, control_stress, DIAGNOSES
from scictrl.synthesize_cpu_diagnostic import PACKS, public_case


def test_blind_payload_has_no_label_or_case_identity():
    for pack in PACKS:
        for clean in (True,False):
            case=public_case(pack,'layer1',clean)
            brief=blind_input(case)
            assert set(brief)=={'objective','proposal'}
            observed=blind_input(case,with_evidence=True)
            assert set(observed)=={'objective','proposal','data','observations'}
            assert observed['observations']==case['state']['observations']


@pytest.mark.parametrize('value',['true','NaN','1.1','-0.1','null','"0.5"'])
def test_bad_probability_rejected(value):
    with pytest.raises(ValueError):
        checked_prediction({'content':'{"issue_probability":'+value+',"evidence":[],"limitations":""}'})


def test_probability_schema():
    assert checked_prediction({'content':'{"issue_probability":0.5,"evidence":[],"limitations":"uncertain"}'})['issue_probability']==.5


def test_control_variants_are_executed(tmp_path):
    rows=control_stress(tmp_path)
    assert len(rows)==32
    assert all(r['validation']['valid'] for r in rows if r['variant'] in {'reference','equivalent_helper'})
    assert all(not r['validation']['valid'] for r in rows if r['variant'] in {'unmodified_flaw','public_output_memorization'})


def test_diagnosis_covers_all_and_does_not_contain_patch():
    assert set(DIAGNOSES)=={p.domain for p in PACKS}
    assert all('def analyze' not in d for d in DIAGNOSES.values())
