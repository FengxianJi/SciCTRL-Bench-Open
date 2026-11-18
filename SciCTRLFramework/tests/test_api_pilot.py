import importlib.util
from pathlib import Path
import pytest
from scictrl.api_backends import json_answer, load_minimax_env
from scictrl.synthesize_cpu_diagnostic import PACKS, public_case

spec=importlib.util.spec_from_file_location('cpu_api_pilot',Path(__file__).resolve().parents[1]/'run_cpu_api_pilot.py')
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_parse_response_strictness():
    assert json_answer('```json\n{"x":1}\n```')=={'x':1}
    with pytest.raises(ValueError):json_answer('[1,2]')


def test_credential_permissions(tmp_path,monkeypatch):
    monkeypatch.delenv('MINIMAX_API_KEY',raising=False)
    p=tmp_path/'key.env'
    p.write_text('MINIMAX_API_KEY=test-placeholder-not-real')
    p.chmod(0o644)
    with pytest.raises(PermissionError):load_minimax_env(p)
    p.chmod(0o600)
    assert load_minimax_env(p)
    monkeypatch.delenv('MINIMAX_API_KEY')


def test_api_loop_does_not_promote_artifact_to_csr(tmp_path,monkeypatch):
    import json
    calls=[]
    source=PACKS[0].reference
    def fake_chat(backend,messages,**kwargs):
        calls.append(messages[-1])
        answer={'action':'experiment' if len(calls)==1 else 'stop', 'source':source,'evidence_report':'test only'}
        return {'content':json.dumps(answer)}
    monkeypatch.setattr(pilot,'chat',fake_chat)
    result=pilot.run(public_case(PACKS[0],'layer2',False),'codex',tmp_path,10)
    assert result['artifact_valid']
    assert result['verified_stop']
    assert result['controlled_success'] is None
    assert 'expected' not in calls[-1]['content']
    assert 'seed' not in calls[-1]['content']


def test_malformed_model_answer_is_not_api_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'chat',lambda *a,**kw:{'content':'not JSON'})
    result=pilot.run(public_case(PACKS[0],'layer2',False),'codex',tmp_path,10)
    assert result['status']=='agent_output_failure'


def test_transport_failure_is_not_scientific_failure(tmp_path,monkeypatch):
    def fail(*a,**kw):raise RuntimeError('offline')
    monkeypatch.setattr(pilot,'chat',fail)
    result=pilot.run(public_case(PACKS[0],'layer2',False),'minimax',tmp_path,10)
    assert result['status']=='infrastructure_or_transport_failure'
    assert result['controlled_success'] is None


def test_existing_measured_checkpoint_counts_as_execution(tmp_path,monkeypatch):
    import json
    case=public_case(PACKS[0],'layer1',True)
    answer={'action':'stop','source':case['state']['analysis_source'],'evidence_report':'inspect supplied measured checkpoint'}
    monkeypatch.setattr(pilot,'chat',lambda *a,**kw:{'content':json.dumps(answer)})
    result=pilot.run(case,'codex',tmp_path,10)
    assert result['artifact_valid'] and result['verified_stop']


def test_initial_checkpoint_hash_mismatch_not_credited():
    case=public_case(PACKS[0],'layer1',True)
    case['state']['observations'][0]['data_hash']='changed'
    assert pilot.initial_verified_source(case) is None
