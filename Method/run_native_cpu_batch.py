"""Bounded native CPU runs. Secrets are received via hidden stdin, never persisted."""
import argparse
from collections import defaultdict
import concurrent.futures
import getpass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--methods',nargs='+',default=['AutoResearchClaw','SAGE','EvoScientist'])
    p.add_argument('--backends',nargs='+',default=['minimax','codex'])
    p.add_argument('--public',type=Path,default=ROOT/'Benchmark/ExecutablePacks/eight_domains_v4/public.jsonl')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--max-calls',type=int,default=64)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--case-order',choices=['file','domain-round-robin'],default='domain-round-robin')
    args=p.parse_args()
    secret=os.environ.pop('MINIMAX_API_KEY','')
    if 'minimax' in args.backends and not secret:secret=getpass.getpass('MiniMax credential (hidden, not saved): ')
    cases=[json.loads(line) for line in args.public.read_text().splitlines()]
    if args.case_order=='domain-round-robin':
        groups=defaultdict(list)
        for case in cases:groups[case['primary_domain']].append(case)
        cases=[group[i] for i in range(max(map(len,groups.values()))) for group in groups.values() if i<len(group)]
    if args.limit:cases=cases[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=[(method,backend,case) for case in cases for method in args.methods for backend in args.backends]
    plan={'methods':args.methods,'backends':args.backends,'case_ids':[c['case_id'] for c in cases],
          'max_calls':args.max_calls,'timeout':args.timeout,'total_jobs':len(jobs),'public':str(args.public.resolve()),
          'case_order':args.case_order,'workers':args.workers}
    plan_path=args.output/'batch_plan.json'
    if plan_path.exists():
        old=json.loads(plan_path.read_text())
        if any(old.get(k)!=plan.get(k) for k in ['methods','backends','case_ids','max_calls','timeout']):
            raise ValueError('existing batch protocol differs; use a new version directory')
    plan_path.write_text(json.dumps(plan,indent=2))
    def run(job):
        method,backend,case=job
        output=(args.output/method/backend/case['case_id']).resolve()
        target=output/'native_result.json'
        if target.exists():return json.loads(target.read_text())
        output.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=''
        if backend=='minimax':env['MINIMAX_API_KEY']=secret
        cmd=[sys.executable,str(ROOT/'Method/run_native_cpu.py'),'--method',method,'--backend',backend,
             '--public',str(args.public.resolve()),'--case-id',case['case_id'],'--output',str(output),'--max-calls',str(args.max_calls)]
        started=time.monotonic()
        with (output/'process.log').open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=log,env=env,cwd=ROOT,start_new_session=True)
            try:proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
        if not target.exists():
            target.write_text(json.dumps({'method':method,'backend':backend,'case_id':case['case_id'],
                'status':'timeout' if time.monotonic()-started>=args.timeout else 'process_crash',
                'returncode':proc.returncode,'native_terminal_reached':False,'full_benchmark_success':None,
                'api_calls':len(list((output/'api_calls').glob('*.json')))},indent=2))
        report=json.loads(target.read_text())
        report['wall_time_s']=round(time.monotonic()-started,2)
        target.write_text(json.dumps(report,indent=2))
        return report
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            result=future.result()
            results.append(result)
            (args.output/'batch_results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:result.get(k) for k in ['method','backend','case_id','status','api_calls','error']}),flush=True)
    print(json.dumps({'complete':True,'runs':len(results)}),flush=True)

if __name__=='__main__':main()
