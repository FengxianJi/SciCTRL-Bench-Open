"""Run public CPU tasks through an API-only integration loop, not a native method.

Public experiment output is returned to the model; held-out validation is never
returned during an attempt. CSR remains unscored pending evidence adjudication.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from scictrl.api_backends import MODELS, chat, json_answer, load_minimax_env
from scictrl.synthesize_cpu_diagnostic import PACKS, digest, execute, validate

SYSTEM='''You are performing a small computational study. Follow the supplied public task.
You may discover that the initial proposal works or needs changes. Preserve the objective and data.
Return a JSON object with action (experiment or stop), source (complete pure Python def analyze(d)),
and evidence_report (your scientific reasoning grounded in actual observations). No markdown.
Use math already supplied in the runtime, not imports. No filesystem, network, GPU or other models.
Supported builtins: sum len min max range zip enumerate sorted reversed set list dict tuple float int abs any all round bool.
Supported methods: math.exp log2 sqrt log isfinite; append pop add items values keys get split sort count index.
Experiment before stopping: stop requires the exact latest successfully executed source.
Experiment output is a measurement, not a correctness grade. Independently check the task objective.
'''

RUNNER_VERSION='cpu-api-loop-1.1'


def initial_verified_source(case):
    source=case['state'].get('analysis_source')
    if source is None: return None
    for observation in reversed(case['state'].get('observations',[])):
        if (observation.get('source_hash')==digest(source) and
            observation.get('data_hash')==digest(case['state']['data'])):
            return digest(source)
    return None


def run(case, backend, output, timeout, *, context=None):
    identity_fields={'case':case,'backend':backend,'model':MODELS[backend],'system':SYSTEM,'runner':RUNNER_VERSION}
    if context is not None:
        identity_fields['additional_context']=context
    identity=digest(identity_fields)
    target=output/backend/(case['case_id']+'.json')
    if target.is_file():
        old=json.loads(target.read_text())
        if old.get('run_identity')==identity and old.get('status')=='completed': return old
        archive=target.parent/'history'/(case['case_id']+'-'+old.get('run_identity','unknown')[:16]+'.json')
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():archive.write_bytes(target.read_bytes())
    trace=[]
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(case)}]
    if context is not None:
        messages.append({'role':'user','content':json.dumps(context)})
    source=''
    last_executed=initial_verified_source(case)
    stopped=False
    started=time.monotonic()
    status='budget_exhausted'
    error=None
    try:
        for step in range(case['budget']['model_decisions']):
            response=chat(backend,messages,timeout=timeout)
            answer=json_answer(response['content'])
            action=answer.get('action')
            source=answer.get('source','')
            record={'step':step,'response':response,'answer':answer}
            trace.append(record)
            messages.append({'role':'assistant','content':response['content']})
            if action=='stop':
                stopped=(last_executed==digest(source))
                status='completed'
                break
            if action!='experiment': raise ValueError('unknown action')
            try:
                actual=execute(source,case['state']['data'])
                last_executed=digest(source)
                observation={'action':'experiment','output':actual,'source_hash':last_executed,
                             'data_hash':digest(case['state']['data'])}
            except Exception as exc:
                observation={'action':'experiment','error':type(exc).__name__+': '+str(exc)[:300]}
            record['observation']=observation
            messages.append({'role':'user','content':json.dumps(observation)})
    except Exception as exc:
        status='agent_output_failure' if isinstance(exc,(ValueError,TypeError,KeyError)) else 'infrastructure_or_transport_failure'
        error=type(exc).__name__+': '+str(exc)[:300]
    pack=next(p for p in PACKS if p.domain==case['primary_domain'])
    verdict=validate(pack,source,version=case['schema_version']) if source else {'valid':False,'checks':[]}
    report={'run_identity':identity,'case_id':case['case_id'],'domain':case['primary_domain'],
            'runner_version':RUNNER_VERSION,
            'public_input':case,
            'additional_context':context,
            'layer':case['layer'],'backend':backend,'model':MODELS[backend],
            'method':'api_integration_loop_NOT_native_AutoResearchClaw_or_other_baseline',
            'status':status,'error':error,'elapsed_s':round(time.monotonic()-started,3),
            'artifact_valid':verdict['valid'],'verified_stop':stopped,
            'evidence_adjudication':'pending','controlled_success':None,'trace':trace,'private_validation':verdict}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--public',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backends',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--minimax-env',type=Path,default=(Path(__file__).resolve().parents[1] / ".secrets" / "minimax.env"))
    parser.add_argument('--limit',type=int)
    parser.add_argument('--domains',nargs='+')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--timeout',type=int,default=180)
    args=parser.parse_args()
    load_minimax_env(args.minimax_env)
    cases=[json.loads(line) for line in args.public.read_text().splitlines() if line.strip()]
    if args.domains: cases=[case for case in cases if case['primary_domain'] in args.domains]
    if args.limit: cases=cases[:args.limit]
    records=[]
    for backend in args.backends:
        import os
        selected=cases[:1] if backend=='minimax' and not os.environ.get('MINIMAX_API_KEY') else cases
        if not 1<=args.workers<=4: raise ValueError('workers must be 1..4')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run,case,backend,args.output,args.timeout) for case in selected]
            for future in as_completed(futures):
                result=future.result()
                records.append({k:v for k,v in result.items() if k not in ['trace','private_validation','public_input']})
                print(backend,result['domain'],result['layer'],result['status'],result['artifact_valid'],flush=True)
                args.output.mkdir(parents=True,exist_ok=True)
                (args.output/'summary.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__': main()
