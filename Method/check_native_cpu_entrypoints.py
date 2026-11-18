"""Record actual entrypoint imports and external runtime prerequisites, not scores."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ENTRIES={
    'AutoResearchClaw':'researchclaw.pipeline.runner',
    'SAGE':'researchclaw.pipeline.mhfa',
    'EvoScientist':'EvoScientist.middleware',
    'AI-Scientist-v2':'ai_scientist.treesearch.agent_manager',
    'AI-Researcher':'research_agent.run_infer_plan',
    'InternAgent-1.5':'internagent.stage',
}

def main():
    folder=ROOT/'Benchmark/Results/native_entrypoint_checks'
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method,module in ENTRIES.items():
        repo=ROOT/'Method'/method
        env=os.environ.copy()
        env.pop('MINIMAX_API_KEY',None)
        env.update(PYTHONPATH=str(repo)+os.pathsep+str(ROOT/'Method/cpu_api_dependencies'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
        guard='import sys\ndef guard(event,args):\n    if event=="subprocess.Popen" and any("pip" in str(x) for x in args[:2]):\n        raise RuntimeError("implicit dependency installation blocked")\nsys.addaudithook(guard)\n'
        command=[sys.executable,'-c',guard+f'import {module}; print("native_import_ok")']
        try:
            p=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True,timeout=120)
            row={'method':method,'entrypoint':module,'returncode':p.returncode,'import_ok':p.returncode==0,
                 'diagnostic':p.stderr[-8000:] if p.returncode else p.stdout[-1000:]}
        except subprocess.TimeoutExpired:row={'method':method,'entrypoint':module,'import_ok':False,'diagnostic':'import timeout (120 s)'}
        row['commit']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        row['full_native_run']=False
        row['classification']='runtime preflight only; not an experiment and not a GPU skip'
        if method=='AI-Researcher':row['docker_available']=shutil.which('docker') is not None
        if method=='InternAgent-1.5':row['upstream_experiment_cli']={name:shutil.which(name) is not None for name in ['claude','iflow']}
        rows.append(row)
        (folder/f'{method}.json').write_text(json.dumps(row,indent=2))
        print(method,row['import_ok'],row['diagnostic'].splitlines()[-1],flush=True)
    (folder/'summary.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
