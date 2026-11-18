"""Render measured API integration diagnostics, never a native-method leaderboard."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import hashlib


def verified_stop(row):
    # Reanalysis only: raw run records remain unchanged. Credit verified input
    # checkpoints as well as new experiments, using source/data hashes.
    if row.get('verified_stop'): return True
    case=row.get('public_input')
    if not case or not row.get('trace'): return False
    final=row['trace'][-1].get('answer',{})
    if final.get('action')!='stop': return False
    def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
    return any(o.get('source_hash')==digest(final.get('source','')) and
               o.get('data_hash')==digest(case['state']['data'])
               for o in case['state'].get('observations',[]))


def render(results: Path,paper: Path,release='v1',output_name='generated_api_pilot.tex'):
    records=[]
    for path in sorted(results.glob('*/*.json')):
        row=json.loads(path.read_text())
        if row.get('method','').startswith('api_integration_loop_'):
            records.append(row)
    grouped=defaultdict(list)
    for row in records: grouped[(row['backend'],row['domain'])].append(row)
    table=['\\begin{table}[h]\\centering\\small',
           '\\caption{API integration diagnostics on the procedural '+release+' pilot; not native-method or controlled-success rates. Unfinished domains are omitted. The v1 contract issues are analyzed separately.}',
           '\\begin{tabular}{llrrr}\\toprule Backend & Domain & Completed & Artifact pass & Verified stop\\\\\\midrule']
    summary=[]
    for (backend,domain),rows in sorted(grouped.items()):
        completed=[r for r in rows if r['status']=='completed']
        valid=sum(r['artifact_valid'] for r in completed)
        stopped=sum(verified_stop(r) for r in completed)
        summary.append({'backend':backend,'domain':domain,'attempt_records':len(rows),
                        'completed':len(completed),'artifact_valid':valid,'verified_stop':stopped,
                        'verification_rule':'recorded experiment OR hashed input checkpoint; reanalysis, raw records preserved'})
        measured_valid=str(valid) if completed else '\\pending'
        measured_stop=str(stopped) if completed else '\\pending'
        table.append(f'{backend} & {domain} & {len(completed)}/{len(rows)} & {measured_valid} & {measured_stop}\\\\')
    table+=['\\bottomrule\\end{tabular}\\end{table}']
    (paper/output_name).write_text('\n'.join(table)+'\n')
    (results/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--paper-dir',type=Path,required=True)
    parser.add_argument('--release',default='v1')
    parser.add_argument('--output-name',default='generated_api_pilot.tex')
    args=parser.parse_args()
    print(json.dumps(render(args.results,args.paper_dir,args.release,args.output_name),indent=2))
