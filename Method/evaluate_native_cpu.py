"""Post-run checks, separate from native controllers and public workspaces.

These are artifact/numerical checks, not a claim of scientific controlled success.
No candidate is selected using hidden scores.
"""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'SciCTRLFramework'))
from scictrl.synthesize_cpu_diagnostic import PACKS,execute,validate,digest,close

def select_source(folder,method):
    if method in {'EvoScientist','AI-Scientist-v2','InternAgent-1.5'}:return folder/'workspace'/'analysis.py'
    # Native final refinement has priority, then native generated experiment.
    for part in ['stage-13/experiment_final','stage-10/experiment']:
        source=folder/part/'analysis.py'
        if source.exists():return source
    return folder/'stage-10/experiment/analysis.py'

def evaluate(folder):
    outcome=json.loads((folder/'native_result.json').read_text())
    case=json.loads((folder/'input_original.json').read_text())
    row={k:outcome.get(k) for k in ['method','backend','case_id','domain','layer','status','api_calls','native_terminal_reached']}
    row.update(numerical_valid=False,full_benchmark_success=None,
        limitation='Artifact validation alone does not establish native execution, evidence attribution, scope compliance, or scientific success.')
    row['native_execution_evidence']=[]
    trace=folder/'native_messages.json'
    if trace.exists():
        for m in json.loads(trace.read_text()):
            if m.get('type')=='tool' and m.get('name')=='execute':
                content=str(m.get('content',''))
                row['native_execution_evidence'].append({'tool_call_id':m.get('tool_call_id'),'content':content,
                    'explicit_success':'Exit code: 0' in content and '[Command failed' not in content})
    row['native_execution_success_observed']=any(e['explicit_success'] for e in row['native_execution_evidence']) if trace.exists() else None
    executions=folder/'native_executions.jsonl'
    if executions.exists():
        row['native_executions']=[json.loads(line) for line in executions.read_text().splitlines()]
        row['native_execution_success_observed']=any(e['exit_code']==0 for e in row['native_executions'])
    source_path=select_source(folder,outcome['method'])
    row['selected_source']=str(source_path.relative_to(folder))
    if not source_path.is_file():
        row['artifact_error']='required final analysis.py absent'
        return row
    source=source_path.read_text()
    row['original_source_hash']=digest(source)
    try:
        tree=ast.parse(source)
        # The evaluator supplies math. Only remove an equivalent, unaliased import.
        filtered=[n for n in tree.body if not (isinstance(n,ast.Import) and len(n.names)==1 and n.names[0].name=='math' and n.names[0].asname is None)]
        adapted=ast.unparse(ast.Module(body=filtered,type_ignores=[]))
        row['normalization']='AST formatting and removal of plain import math only; no repair synthesis'
        row['public_execution']=execute(adapted,case['state']['data'])
        row['matched_native_scientific_execution']=False
        for execution in row.get('native_executions',[]):
            try:observed=json.loads(execution['output'])
            except (TypeError,ValueError):continue
            if (execution['exit_code']==0 and not execution.get('truncated')
                and execution.get('source_hash_before')==row['original_source_hash']
                and execution.get('data_hash_before')==digest(case['state']['data'])
                and close(observed,row['public_execution'])):
                row['matched_native_scientific_execution']=True
        pack=next(p for p in PACKS if p.domain==case['primary_domain'])
        row['heldout']=validate(pack,adapted,version=case['schema_version'])
        row['numerical_valid']=row['heldout']['valid']
        row['source_changed']=source!=case['state']['analysis_source']
    except Exception as exc:row['artifact_error']=type(exc).__name__+': '+str(exc)[:500]
    resource=folder/('input' if outcome['method'] in {'AutoResearchClaw','SAGE'} else 'workspace')/'data.json'
    row['supplied_data_unchanged']=resource.exists() and json.loads(resource.read_text())==case['state']['data']
    return row

def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    args=p.parse_args()
    rows=[]
    for path in sorted(args.output.glob('*/*/*/native_result.json')):
        if not (path.parent/'input_original.json').exists():continue
        row=evaluate(path.parent)
        (path.parent/'artifact_evaluation.json').write_text(json.dumps(row,indent=2))
        rows.append(row)
    summary={'runs':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
             'numerical_valid':sum(r['numerical_valid'] for r in rows),'native_terminal':sum(bool(r['native_terminal_reached']) for r in rows),
             'full_benchmark_success':None,'rows':rows}
    (args.output/'artifact_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
