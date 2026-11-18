import json
import pytest
from scictrl.synthesize_cpu_diagnostic import PACKS, RELEASE_DIR, check_source, close, execute, generate, instance, public_case, validate


@pytest.mark.parametrize('pack', PACKS, ids=lambda p:p.domain)
def test_real_reference_and_flawed_contrast(pack):
    assert validate(pack,pack.reference)['valid']
    assert not any(r['valid'] for r in validate(pack,pack.flawed)['checks'])


@pytest.mark.parametrize('pack', PACKS, ids=lambda p:p.domain)
def test_start_layers_and_hashes(pack):
    local=public_case(pack,'layer1',False)
    idea=public_case(pack,'layer2',False)
    assert local['state']['observations'][0]['output']==execute(pack.flawed,local['state']['data'])
    assert idea['state']['analysis_source'] is None
    assert idea['state']['observations']==[]
    assert local['case_id']!=idea['case_id']
    for secret in ['reference_source','revision_role','clean','expected','k']:
        assert secret not in local


@pytest.mark.parametrize('source',[
    'import os\ndef analyze(d): return 1',
    'def analyze(d): return open("/etc/passwd").read()',
    'def analyze(d): return d.__class__',
    'def analyze(d): return eval("1")',
    'def analyze(d): return getattr(d,"items")()',
])
def test_reject_unsafe_analysis(source):
    with pytest.raises(ValueError): check_source(source)


def test_type_and_length_checks():
    assert not close(0.0001,0)
    assert not close(True,1)
    assert not close(float('nan'),1)
    assert not close([1],[1,2])
    assert not close({'score':1},1)


def test_equivalent_solution_not_exact_patch():
    pack=PACKS[0]
    source='def analyze(d):\n    out=[]\n    for r,t in d["experiments"]:\n        out.append(math.exp(-t*r))\n    return out\n'
    assert source!=pack.reference
    assert validate(pack,source)['valid']


def test_public_holdout_not_in_input():
    for pack in PACKS:
        data,_=instance(pack.domain,7)
        hidden,_=instance(pack.domain,101)
        assert data!=hidden


def test_generation_is_separate_and_complete(tmp_path):
    result=generate(tmp_path)
    assert result['public_cases']==32
    for layer in ['Layer1','Layer2']:
        rows=[json.loads(x) for x in (tmp_path/layer/RELEASE_DIR/'public.jsonl').read_text().splitlines()]
        assert len(rows)==16
        assert len({r['primary_domain'] for r in rows})==8


def test_all_private_shapes_are_explicit_in_public_contract():
    for pack in PACKS:
        for layer in ['layer1','layer2']:
            case=public_case(pack,layer,False)
            _,expected=instance(pack.domain,101)
            kind='array' if isinstance(expected,list) else 'number'
            assert case['contract']['return_schema']['type']==kind


def test_pure_helper_function_is_not_an_illegal_revision():
    source='def mean(xs):\n    return sum(xs)/len(xs)\ndef analyze(d):\n    return mean([1,2,3])\n'
    assert execute(source,{})==2


def test_recursive_negation_contract_and_reference_agree():
    pack=next(p for p in PACKS if p.domain=='NLP')
    assert 'two not tokens cancel' in pack.objective
    assert execute(pack.reference,{'texts':['not good','not not good','not not not poor']})==[-1,1,1]


def test_vision_holdouts_do_not_share_the_public_answer_order():
    _,public=instance('vision',7)
    assert all(instance('vision',seed)[1]!=public for seed in (101,202,303))
    # Old releases remain reproducible, including their documented weakness.
    assert instance('vision',7,version='cpu-pilot-1.2')[1]==instance('vision',101,version='cpu-pilot-1.2')[1]
