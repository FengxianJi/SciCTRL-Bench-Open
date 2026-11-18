"""Native compatibility inventory; block import-triggered pip installation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

METHODS={
 'AutoResearchClaw': ('https://github.com/aiming-lab/AutoResearchClaw','researchclaw.pipeline.executor','researchclaw/prompts',[]),
 'SAGE': ('https://github.com/JieMaMagic/SAGE','researchclaw.pipeline.mhfa','prompts.default.yaml',[]),
 'AI-Scientist-v2': ('https://github.com/SakanaAI/AI-Scientist-v2','ai_scientist.llm','ai_scientist/treesearch',['CPU adaptation not verified; upstream CUDA-oriented launcher']),
 'EvoScientist': ('https://github.com/EvoScientist/EvoScientist','EvoScientist.llm.models','EvoScientist/prompts.py',['native graph/provider adaptation pending']),
 'AI-Researcher': ('https://github.com/HKUDS/AI-Researcher','research_agent.run_infer_plan','research_agent/inno/agents/inno_agent',['upstream container workflow requires Docker; unavailable here']),
 'InternAgent-1.5': ('https://github.com/InternScience/InternAgent','internagent.stage','internagent/prompts.py',['disable local embeddings and non-selected inference backends; native mapping pending']),
}


def inspect(root,python):
    rows=[]
    for name,(url,module,prompts,notes) in METHODS.items():
        repo=root/name
        commit=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
        deps=root / "cpu_api_dependencies"
        env=dict(os.environ,PYTHONPATH=str(repo.resolve())+os.pathsep+str(deps),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
        guard='import sys\ndef guard(event,args):\n    if event == "subprocess.Popen" and any("pip" in str(x) for x in args[:2]):\n        raise RuntimeError("import-triggered pip installation blocked; provision dependencies explicitly")\nsys.addaudithook(guard)\n'
        check=subprocess.run([str(python),'-c',guard+f'import {module}; print("native_import_ok")'],cwd=repo,
                             env=env,text=True,capture_output=True,timeout=60)
        path=repo/prompts
        files=list(path.rglob('*.py')) if path.is_dir() else [path]
        hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
        # Dependency errors only, not environment dumps or complete remote responses.
        diagnostic=(check.stderr.strip().splitlines()[-1] if check.returncode else 'native_import_ok')
        row={'method':name,'official_repository':url,'local_path':str(repo.resolve()),
             'commit':commit.stdout.strip() if commit.returncode==0 else None,
             'import_ok':check.returncode==0,'diagnostic':diagnostic,
             'prompt_hashes':hashes,'notes':notes,
             'full_reproduction':'pending','cpu_only':True,
             'active_models':['MiniMax-M3','gpt-5.6-sol'],'fallback_models':[]}
        rows.append(row)
        print(name,diagnostic,flush=True)
    (root/'reproduction_manifest.json').write_text(json.dumps(rows,indent=2))
    deps=root / "cpu_api_dependencies"
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(deps)])}
    (root/'cpu_api_dependency_lock.json').write_text(json.dumps(versions,indent=2,sort_keys=True))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    args=parser.parse_args()
    inspect(Path(__file__).resolve().parent,args.python.resolve())
