import io
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Method'))
from native_bridge import Bridge,NativeBudgetExceeded

def test_minimax_real_wire_contract_and_secret_redaction(tmp_path,monkeypatch):
    captured={}
    def open_request(request,timeout):
        captured.update(json.loads(request.data))
        return io.StringIO(json.dumps({'model':'MiniMax-M3','choices':[{'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}],'usage':{'total_tokens':2}}))
    monkeypatch.setattr('urllib.request.urlopen',open_request)
    bridge=Bridge('minimax','test-credential-never-log',tmp_path,1)
    assert bridge.completion([{'role':'user','content':'test'}])['content']=='ok'
    assert captured['thinking']=={'type':'disabled'}
    assert captured['model']=='MiniMax-M3'
    assert 'test-credential-never-log' not in (tmp_path/'0000.json').read_text()
    with pytest.raises(NativeBudgetExceeded):bridge.completion([])

def test_rejects_wrong_model(tmp_path,monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:io.StringIO(json.dumps({'model':'other-model'})))
    with pytest.raises(RuntimeError,match='unexpected model'):Bridge('minimax','test-key',tmp_path).completion([])

def test_public_transport_does_not_import_private_evaluation():
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ['SciCTRLFramework','Benchmark']))
    script='import sys; from scictrl.api_backends import chat; from agent_adapters.cpu_state import render_cpu_state; assert "scictrl.synthesize_cpu_diagnostic" not in sys.modules; assert "scictrl.evaluator" not in sys.modules'
    p=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr

def test_public_hash_remains_compatible():
    from scictrl.public_contract import digest as public_digest
    from scictrl.synthesize_cpu_diagnostic import digest as private_digest
    assert public_digest({'x':[1,2.5]})==private_digest({'x':[1,2.5]})
