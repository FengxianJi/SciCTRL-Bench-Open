"""Finish reporting after the already-launched native batches end.

No model calls and no credentials. Incomplete batches stay explicitly incomplete.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'native_cpu_v2':128,'native_evo_cpu_v1':64}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--max-wait-s',type=int,default=28800)
    args=p.parse_args()
    start=time.monotonic()
    output=ROOT/'Benchmark/Results/native_progress.json'
    while True:
        counts={name:len(list((ROOT/'Benchmark/Results'/name).glob('*/*/*/native_result.json'))) for name in EXPECTED}
        complete=all(counts[k]>=v for k,v in EXPECTED.items())
        expired=time.monotonic()-start>=args.max_wait_s
        status={'timestamp_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'state':'batches_ended' if complete else 'wait_deadline' if expired else 'running',
                'completed_attempts':counts,'expected_attempts':EXPECTED,
                'scientific_leaderboard_complete':False}
        output.write_text(json.dumps(status,indent=2))
        if complete or expired:break
        time.sleep(30)
    report=subprocess.run([sys.executable,str(ROOT/'Method/summarize_native_runs.py')],cwd=ROOT,capture_output=True,text=True)
    paper=ROOT/'_ICLR27_SciCTRL_Bench__A_Benchmark__For_End_to_End_AI_Scientists_Controlled_Feedback_Revision_Evaluation'
    compilation=[]
    for _ in range(2):
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','iclr2027_conference.tex'],cwd=paper,capture_output=True,text=True,timeout=180)
        compilation.append(result.returncode)
        if result.returncode:break
    status.update(report_exit_code=report.returncode,latex_exit_codes=compilation,
        note='Only the already-launched ARC/SAGE/Evo batches are finalized; blocked or partial methods are not declared reproduced.')
    output.write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':main()
