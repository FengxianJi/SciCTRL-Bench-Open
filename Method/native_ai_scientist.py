"""AI Scientist-v2's actual four-stage experiment manager on CPU-only inputs."""
import json
import os
from pathlib import Path
import sys

def run(args,case,workspace,bridge):
    from native_http_bridge import serve
    from native_bridge import NativeBudgetExceeded
    from agent_adapters.cpu_state import render_cpu_state
    cpu_deps=Path(__file__).resolve().parent / "ai_scientist_cpu_dependencies"
    sys.path.insert(0,str(cpu_deps))
    import torch
    if torch.cuda.is_available():raise RuntimeError('CPU-only admission failed')
    from omegaconf import OmegaConf
    from ai_scientist.treesearch.agent_manager import AgentManager
    from ai_scientist.treesearch.interpreter import Interpreter
    import ai_scientist.treesearch.parallel_agent as parallel
    import ai_scientist.treesearch.backend.backend_openai as backend_openai
    import openai
    original_query=backend_openai.query
    def pickle_safe_query(*a,**k):
        try:return original_query(*a,**k)
        except openai.APIStatusError as exc:
            raise RuntimeError(f'native provider request failed: HTTP {exc.status_code}: {exc.message[:200]}') from None
    backend_openai.query=pickle_safe_query
    repo=Path(__file__).resolve().parent / "AI-Scientist-v2"
    os.environ['AI_SCIENTIST_ROOT']=str(repo)
    # Upstream nvidia-smi probe ignores an empty CUDA_VISIBLE_DEVICES.
    # Disable its allocation, not the native experiment/search controller.
    parallel.get_gpu_count=lambda:0
    work=workspace/'workspace'
    brief=render_cpu_state(case,work)
    brief+='\nCPU only. Use the supplied task and data; do not substitute a new study. '
    brief+='Write analysis.py with the contracted pure analyze(d), run it, and save result.json and report.md. '
    brief+='No GPU, local language model, package installation, new dataset download, or publication. '
    brief+='Generated main scripts must call analyze(d) and print measured results, not merely define functions. '
    task={'Title':case['objective'],'Abstract':brief,'Short Hypothesis':case['state']['proposal'],
          'Experiments':[brief],'Risk Factors and Limitations':json.dumps(case['contract'])}
    if case['state']['analysis_source']:task['Code']=case['state']['analysis_source']
    cfg=OmegaConf.load(repo/'bfts_config.yaml')
    cfg.workspace_dir=str(work)
    cfg.log_dir=str(workspace/'logs')
    cfg.data_dir=str(work)
    cfg.desc_file=str(work/'public_state.json')
    cfg.exp_name=case['case_id']
    cfg.exec.timeout=30
    cfg.agent.num_workers=1
    cfg.agent.search.num_drafts=1
    cfg.agent.multi_seed_eval.num_seeds=1
    for key in cfg.agent.stages:cfg.agent.stages[key]=2
    for name in ['code','feedback','vlm_feedback']:
        cfg.agent[name].model=bridge.model
        cfg.agent[name].max_tokens=8192
    cfg.agent.summary={'model':bridge.model,'temp':1.0}
    cfg.agent.select_node={'model':bridge.model,'temp':1.0}
    cfg.report.model=bridge.model
    (workspace/'logs').mkdir(exist_ok=True)
    (workspace/'native_settings.yaml').write_text(OmegaConf.to_yaml(cfg))
    (workspace/'native_task.json').write_text(json.dumps(task,indent=2))
    interpreter=Interpreter(work,timeout=30,format_tb_ipython=False)
    def checkpoint(stage,journal):
        rows=[{'id':n.id,'code':n.code,'is_buggy':n.is_buggy,'term_out':n.term_out,
               'metric':str(n.metric)} for n in journal.nodes]
        (workspace/'logs'/f'{stage.name}.json').write_text(json.dumps(rows,indent=2,default=str))
        if bridge.count>=bridge.max_calls:raise NativeBudgetExceeded('native model-call budget exhausted')
    with serve(bridge) as (url,token):
        os.environ['OPENAI_BASE_URL']=url
        os.environ['OPENAI_API_KEY']=token  # Only a scoped loopback token, not a provider key.
        try:
            manager=AgentManager(json.dumps(task),cfg,work)
            manager.run(exec_callback=interpreter.run,step_callback=checkpoint)
            completed=list(manager.completed_stages)
            return {'native_terminal_reached':any(str(s).startswith('4_') for s in completed),
                    'completed_stages':completed,'native_component':'four-stage experiment manager; paper launcher not run',
                    'torch_version':torch.__version__,'torch_cuda_available':False}
        finally:
            interpreter.cleanup_session()
            os.environ.pop('OPENAI_API_KEY',None)
            os.environ.pop('OPENAI_BASE_URL',None)
