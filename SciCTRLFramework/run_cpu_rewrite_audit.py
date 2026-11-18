"""API rewrite + semantic/surface audit; proposals never auto-promote to main data."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from scictrl.api_backends import MODELS,chat,json_answer,load_minimax_env
from scictrl.synthesize_cpu_diagnostic import digest

REWRITE='''Rewrite ONLY the proposal as a coherent scientific study description.
Preserve every proposed methodological choice, even if questionable. Do not repair,
criticize, identify a flaw, or add restrictions. Do not alter the task objective or data.
Return JSON with proposal (string). This is a task-authoring draft, not scientific approval.'''
AUDIT='''Compare the original and rewritten scientific proposals against the public task.
Return JSON with semantic_preserved (boolean), changed_choices (list),
surface_issue_detectable (boolean), evidence (list of quoted proposal passages),
and review_notes (string). Identify uncertainty. Do not assume there must be an error.
This review has no private construction label or reference solution.'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--public',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backends',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--limit',type=int,default=1)
    args=parser.parse_args()
    load_minimax_env((Path(__file__).resolve().parents[1] / ".secrets" / "minimax.env"))
    cases=[json.loads(x) for x in args.public.read_text().splitlines()][:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    for backend in args.backends:
        for case in cases:
            record={'case_id':case['case_id'],'backend':backend,'source_hash':digest(case),
                    'admission':'not_admitted_requires_independent_execution_and_expert_review'}
            try:
                reply=chat(backend,[{'role':'system','content':REWRITE},{'role':'user','content':json.dumps(case)}])
                proposal=json_answer(reply['content'])['proposal']
                if not isinstance(proposal,str) or not proposal.strip(): raise ValueError('invalid rewrite')
                candidate=deepcopy(case)
                candidate['state']['proposal']=proposal
                audit=chat(backend,[{'role':'system','content':AUDIT},
                    {'role':'user','content':json.dumps({'original':case,'rewritten':candidate})}])
                record.update({'status':'completed','rewrite_response':reply,'candidate':candidate,
                               'audit_response':audit,'audit':json_answer(audit['content']),
                               'note':'single-judge smoke only; not blind balanced discrimination statistics'})
            except Exception as error:
                record.update({'status':'failed','error':type(error).__name__+': '+str(error)[:300]})
            (args.output/(backend+'-'+case['case_id']+'.json')).write_text(json.dumps(record,indent=2))
            print(backend,record['status'],flush=True)


if __name__=='__main__':main()
