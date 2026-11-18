"""Run actual ARC/SAGE pipelines or EvoScientist graph on public CPU states.

No private annotation is loaded into the agent process. Evaluation is separate.
Native budget overrides are explicit compatibility settings, not the old six-call pilot.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value,indent=2))


def researchclaw(args,case,workspace,bridge):
    import yaml
    from researchclaw.config import RCConfig
    from researchclaw.adapters import AdapterBundle
    from researchclaw.llm.client import LLMClient,LLMResponse,LLMConfig
    from researchclaw.pipeline.runner import execute_pipeline
    from researchclaw.pipeline.stages import Stage
    from agent_adapters.cpu_state import render_cpu_state
    brief=render_cpu_state(case,workspace/'input')
    brief+='\n\nRun CPU-only. The supplied study is the task; do not replace its objective or data with a different study. '
    brief+='Implement the requested pure analyze(d) in analysis.py, execute it on the supplied data, and include measured outputs in your report. '
    brief+='No GPU, external LLM, publication upload, or new dataset download is permitted. '
    brief+='Research novelty is not a requirement. Provided proposal is the starting idea, not a newly generated hypothesis. '
    brief+='Use stdlib/numpy for driver code; analyze(d) itself must follow the public pure-Python contract.'
    brief+=' The native executable entry is main.py: it must call analyze(d) and print measured outputs, not merely define functions. '
    brief+='The supplied data file is '+str(workspace/'input'/'data.json')+'. '
    brief+='Do not invent scalar metrics to pass the native paper-writing gate.'
    write(workspace/'adapter_context'/'goal.md',brief)
    write(workspace/'adapter_context'/'hypotheses.md','# Provided study proposal (adapter mapping)\n'+case['state']['proposal']+'\n\nObjective: '+case['objective'])
    plan={'objectives':[case['objective']],'datasets':[{'name':'supplied-public-data','data':case['state']['data']}],
          'proposed_methods':[case['state']['proposal']],'metrics':[case['contract']['numerical_acceptance']],
          'provenance':'adapter mapping of public task, not an agent-created plan'}
    if case['layer']=='layer1':
        write(workspace/'stage-09'/'exp_plan.yaml',yaml.safe_dump(plan))
        write(workspace/'stage-10'/'experiment'/'analysis.py',case['state']['analysis_source'])
        write(workspace/'stage-10'/'experiment'/'data.json',case['state']['data'])
        write(workspace/'stage-10'/'experiment'/'main.py',
              'import json\nimport math\nimport analysis\nanalysis.math = math\n'
              'with open("data.json") as f:\n    data = json.load(f)\n'
              'print(json.dumps(analysis.analyze(data)))\n')
        for i,obs in enumerate(case['state']['observations']):
            write(workspace/'stage-12'/'runs'/f'checkpoint-{i}.json',
                  {'provenance':'real supplied checkpoint; no fabricated metric','returncode':0,'status':'completed',
                   'stdout':json.dumps(obs['output']),'stderr':'','metrics':{},'observation':obs})
    settings={
        'project':{'name':'scictrl-native-cpu','mode':'full-auto','profile':'ml'},
        'research':{'topic':brief,'domains':[case['primary_domain']]},
        'runtime':{'timezone':'UTC','retry_limit':0},'notifications':{'channel':'local'},
        'knowledge_base':{'backend':'markdown','root':str(workspace/'kb')},
        'llm':{'provider':'openai-compatible','base_url':'https://api.minimaxi.com/v1','api_key_env':'NATIVE_TRANSPORT_KEY',
               'primary_model':bridge.model,'fallback_models':[]},
        'experiment':{'mode':'sandbox','time_budget_sec':30,'max_iterations':2,'max_refine_duration_sec':120,
            'sandbox':{'gpu_required':False,'python_path':sys.executable,'allowed_imports':['math','random','json','csv','numpy'],'max_memory_mb':1024},
            'benchmark_agent':{'enabled':False},'figure_agent':{'enabled':False,'nano_banana_enabled':False},
            'repair':{'enabled':True,'max_cycles':1,'use_opencode':False,'timeout_sec_per_cycle':60},
            'cli_agent':{'provider':'llm'},'opencode':{'enabled':False}},
        'web_search':{'enabled':False},'openclaw_bridge':{},'security':{'allow_publish_without_approval':False},
    }
    if args.method=='SAGE':settings['mhfa']={'enabled':True,'generation':{'gen_model':bridge.model},'critic':{'critic_model':bridge.model},'judge':{'gen_model':bridge.model,'judge_model':bridge.model}}
    config=RCConfig.from_dict(settings,check_paths=False)
    write(workspace/'native_settings.json',settings)
    class Transport:
        config=LLMConfig(base_url='native-bridge',api_key='in-memory-transport',primary_model=bridge.model,fallback_models=[])
        def chat(self,messages,*,system=None,**kwargs):
            req=([{'role':'system','content':system}] if system else [])+messages
            r=bridge.completion(req,max_tokens=min(kwargs.get('max_tokens') or 8192,16000))
            finish=r.get('_transport_finish_reason','stop')
            return LLMResponse(content=r.get('content') or '',model=bridge.model,finish_reason=finish,
                truncated=finish=='length',raw=r)
    LLMClient.from_rc_config=classmethod(lambda cls,c:Transport())
    class UnavailableWeb:
        def fetch(self,url):raise RuntimeError('web access disabled for public-resource CPU task')
        def open(self,url):raise RuntimeError('browser disabled for public-resource CPU task')
    adapters=AdapterBundle()
    adapters.web_fetch=UnavailableWeb()
    adapters.browser=UnavailableWeb()
    # Native input lookup walks stage-*; this directory contains only mapped public text.
    context=workspace/'stage-00'
    write(context/'goal.md',brief)
    write(context/'hypotheses.md',(workspace/'adapter_context'/'hypotheses.md').read_text())
    write(context/'synthesis.md','# Supplied public task context (adapter mapping, not a literature review)\n\n'+brief)
    # Original run files, stage executor, refinement and decision routing stay native.
    results=execute_pipeline(run_dir=workspace,run_id=case['case_id'],config=config,adapters=adapters,
        from_stage=Stage.RESULT_ANALYSIS if case['layer']=='layer1' else Stage.EXPERIMENT_DESIGN,
        to_stage=Stage.CITATION_VERIFY,auto_approve_gates=True,stop_on_gate=False,kb_root=workspace/'kb')
    return {'native_terminal_reached':bool(results and int(results[-1].stage)==23 and results[-1].status.value=='done'),
            'stages':[{'stage':r.stage.name,'status':r.status.value,'error':r.error,'decision':r.decision} for r in results]}


def evo(args,case,workspace,bridge):
    for key,part in [('EVOSCIENTIST_DATA_DIR','runtime'),('EVOSCIENTIST_MEMORIES_DIR','memories'),
                     ('EVOSCIENTIST_SKILLS_DIR','skills'),('EVOSCIENTIST_WORKSPACE_DIR','workspace')]:
        os.environ[key]=str(workspace/part)
    # Prevent importing a user's legacy sessions or external MCP configuration.
    write(workspace/'runtime'/'.migrated','isolated benchmark workspace\n')
    from agent_adapters.cpu_state import render_cpu_state
    from native_bridge import langchain_model
    from EvoScientist.config.settings import EvoScientistConfig
    from EvoScientist.EvoScientist import create_cli_agent
    import importlib
    graph_module=importlib.import_module('EvoScientist.EvoScientist')
    from EvoScientist.backends import CustomSandboxBackend
    import threading
    execution_lock=threading.Lock()
    original_execute=CustomSandboxBackend.execute
    def observed_execute(backend,command,**kwargs):
        from scictrl.public_contract import digest
        source=work/'analysis.py'
        data=work/'data.json'
        source_hash=digest(source.read_text()) if source.exists() else None
        try:data_hash=digest(json.loads(data.read_text())) if data.exists() else None
        except ValueError:data_hash=None
        response=original_execute(backend,command,**kwargs)
        with execution_lock:
            with (workspace/'native_executions.jsonl').open('a') as log:
                log.write(json.dumps({'command':command,'exit_code':response.exit_code,
                    'output':response.output,'truncated':response.truncated,
                    'source_hash_before':source_hash,'data_hash_before':data_hash})+'\n')
        return response
    CustomSandboxBackend.execute=observed_execute
    import EvoScientist.mcp.client as mcp_client
    mcp_client.USER_CONFIG_DIR=workspace/'runtime'/'config'
    mcp_client.USER_MCP_CONFIG=mcp_client.USER_CONFIG_DIR/'mcp.yaml'
    model=langchain_model(bridge)
    config=EvoScientistConfig(model=bridge.model,provider='openai',auxiliary_model=bridge.model,auxiliary_provider='openai',
        auto_mode=True,auto_approve=True,enable_async_subagents=False,enable_scheduler=False,
        recursion_limit=160,sandbox_execute_timeout=30,model_fallbacks='',memory_workers_enabled=False,
        memory_skill_synthesis_enabled=False)
    # All native subagents and helper calls use the selected backbone.
    graph_module._chat_model=model
    graph_module._chat_model_key=(bridge.model,'openai')
    graph_module._auxiliary_chat_model=model
    graph_module._auxiliary_chat_model_key=(bridge.model,'openai')
    import EvoScientist.llm.models as native_models
    native_models.get_chat_model=lambda *a,**k:model
    work=workspace/'workspace'
    brief=render_cpu_state(case,work)
    brief+='\n\nComplete this supplied computational study, not a new research topic. '
    brief+='The code-generation mode is Lite. Use the supplied local data.json. Write the final pure analyze(d) function in analysis.py, '
    brief+='execute it with math available, save measured output to result.json and a scientific report to report.md. '
    brief+='CPU only; no GPU, third-party model, package installation, dataset download or publication. '
    brief+='Keep native planning, experiment and revision tools. Do not access files outside this task workspace. '
    brief+='All evidence must come from the supplied task resources or executed calculations. '
    brief+='File tools use virtual paths such as /analysis.py, but shell execute starts in the physical task workspace: '
    brief+='use relative paths analysis.py and data.json in shell commands, not /analysis.py or /data.json. '
    brief+='Use python for execution. A successful shell execution is required; writing inferred numbers to result.json is not execution. '
    brief+='Do not schedule background jobs. The model-call budget is '+str(args.max_calls)+'.'
    graph=create_cli_agent(workspace_dir=str(work),config=config,chat_model=model)
    result=graph.invoke({'messages':[{'role':'user','content':brief}]},config={'configurable':{'thread_id':case['case_id']},'recursion_limit':160})
    write(workspace/'native_messages.json',[m.model_dump(mode='json') for m in result.get('messages',[])])
    messages=result.get('messages',[])
    terminal=bool(messages and messages[-1].type=='ai' and not getattr(messages[-1],'tool_calls',None) and not result.get('__interrupt__'))
    return {'native_terminal_reached':terminal,'native_message_count':len(messages)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['AutoResearchClaw','SAGE','EvoScientist','AI-Scientist-v2','InternAgent-1.5'],required=True)
    p.add_argument('--backend',choices=['minimax','codex'],required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--case-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-calls',type=int,default=64)
    args=p.parse_args()
    key=os.environ.pop('MINIMAX_API_KEY','')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    sys.path[:0]=[str(ROOT/'Method'/args.method),str(ROOT/'Method/cpu_api_dependencies'),str(ROOT/'SciCTRLFramework'),str(ROOT/'Benchmark')]
    from native_bridge import Bridge,NativeBudgetExceeded
    from scictrl.public_contract import digest
    original=next(json.loads(l) for l in args.public.read_text().splitlines() if json.loads(l)['case_id']==args.case_id)
    case=deepcopy(original)
    case['budget']['model_decisions']=args.max_calls
    workspace=args.output.resolve()
    workspace.mkdir(parents=True,exist_ok=True)
    snapshots=['Method/run_native_cpu.py','Method/native_bridge.py','Method/native_http_bridge.py',
        'Method/native_ai_scientist.py','Method/native_anthropic_bridge.py','Method/native_internagent.py',
        'SciCTRLFramework/scictrl/api_backends.py','SciCTRLFramework/scictrl/public_contract.py','Benchmark/agent_adapters/cpu_state.py']
    for relative in snapshots:
        write(workspace/'adapter_snapshot'/relative,(ROOT/relative).read_text())
    write(workspace/'input_original.json',original)
    write(workspace/'input_effective.json',case)
    bridge=Bridge(args.backend,key,workspace/'api_calls',args.max_calls)
    report={'method':args.method,'backend':args.backend,'case_id':case['case_id'],'domain':case['primary_domain'],'layer':case['layer'],
            'adapter_source_hash':digest(Path(__file__).read_text()),
            'original_input_hash':digest(original),'effective_input_hash':digest(case),'max_calls':args.max_calls,
            'budget_override':True,'scope':'native-controller CPU compatibility run, not the six-call shared-loop pilot',
            'cpu_only':True,'full_benchmark_success':None}
    try:
        if args.method=='InternAgent-1.5':
            from native_internagent import run
            outcome=run(args,case,workspace,bridge)
        elif args.method=='AI-Scientist-v2':
            from native_ai_scientist import run
            outcome=run(args,case,workspace,bridge)
        else:
            outcome=evo(args,case,workspace,bridge) if args.method=='EvoScientist' else researchclaw(args,case,workspace,bridge)
        report.update(outcome,status='completed' if outcome['native_terminal_reached'] else 'native_process_failure')
    except NativeBudgetExceeded as exc:
        report.update(status='budget_exhausted',error=str(exc),native_terminal_reached=False)
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc)[:500],native_terminal_reached=False)
        write(workspace/'traceback.txt',traceback.format_exc().replace(key,'[REDACTED]') if key else traceback.format_exc())
    report['api_calls']=bridge.count
    report['private_pack_module_loaded']='scictrl.synthesize_cpu_diagnostic' in sys.modules
    write(workspace/'native_result.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
