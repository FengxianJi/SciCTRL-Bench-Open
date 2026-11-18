"""Native ARC/SAGE topic-stage smoke with real API transport, no GPU execution.

This validates native prompt + stage integration, NOT a full research trajectory.
Run in separate processes because ARC and SAGE use the same Python package name.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['AutoResearchClaw','SAGE'],required=True)
    parser.add_argument('--backend',choices=['minimax','codex'],required=True)
    parser.add_argument('--public',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    repo=root/'Method'/args.method
    sys.path[:0]=[str(repo),str(root/'SciCTRLFramework'),str(root/'Benchmark')]
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from scictrl.api_backends import MODELS,chat,load_minimax_env
    from agent_adapters.cpu_state import render_cpu_state
    from researchclaw.config import RCConfig
    from researchclaw.adapters import AdapterBundle
    from researchclaw.llm.client import LLMClient,LLMResponse
    from researchclaw.pipeline.executor import execute_stage
    from researchclaw.pipeline.stages import Stage
    import researchclaw
    if Path(researchclaw.__file__).resolve().parent != (repo/'researchclaw').resolve():
        raise RuntimeError('wrong native method imported')
    load_minimax_env(root/'.secrets/minimax.env')
    cases=[json.loads(line) for line in args.public.read_text().splitlines()]
    case=next(c for c in cases if c['layer']=='layer2')
    args.output=args.output.resolve()
    topic=render_cpu_state(case,args.output)
    config=RCConfig.from_dict({
        'project':{'name':'scictrl-native-stage-smoke','mode':'full-auto'},
        'research':{'topic':topic,'domains':[case['primary_domain']]},
        'runtime':{'timezone':'UTC'},'notifications':{'channel':'local'},
        'knowledge_base':{'backend':'markdown','root':str(args.output/'kb')},
        'llm':{'provider':'openai-compatible','base_url':'https://api.minimaxi.com/v1',
               'api_key_env':'MINIMAX_API_KEY','primary_model':MODELS[args.backend],'fallback_models':[]},
        'experiment':{'mode':'sandbox','time_budget_sec':30,'max_iterations':1,
                      'sandbox':{'gpu_required':False,'python_path':sys.executable}},
        'web_search':{'enabled':False},'openclaw_bridge':{},
    },check_paths=False)
    calls=[]
    class Transport:
        config=SimpleNamespace(base_url='api-transport',api_key='transport-managed')
        def chat(self,messages,*,system=None,**kwargs):
            request=([{'role':'system','content':system}] if system else [])+messages
            result=chat(args.backend,request,timeout=180,max_tokens=min(kwargs.get('max_tokens') or 4096,6000))
            calls.append({'request':request,'response':result})
            (args.output/'api_calls.json').write_text(json.dumps(calls,indent=2))
            return LLMResponse(content=result['content'],model=result['returned_model'],finish_reason='stop')
    # Replace transport only. Native stage executor, prompts, contracts remain unchanged.
    LLMClient.from_rc_config=classmethod(lambda cls,c:Transport())
    result=execute_stage(Stage.TOPIC_INIT,run_dir=args.output,run_id=case['case_id'],
                         config=config,adapters=AdapterBundle(),auto_approve_gates=True)
    report={'method':args.method,'backend':args.backend,'stage':'TOPIC_INIT','status':result.status.value,
            'error':result.error,'api_calls':len(calls),'full_reproduction':False,
            'scope':'native topic-stage API smoke only; no experiment or revision assessed'}
    (args.output/'native_smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
