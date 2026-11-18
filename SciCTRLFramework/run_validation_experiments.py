"""Registered CPU construction, blind-review and diagnosis-given experiments.

These evaluate the diagnostic release, never impersonate a native scientist.
Private labels are joined only after each independent model response is saved.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
from pathlib import Path
import random

from scictrl.api_backends import MODELS, chat, json_answer, load_minimax_env
from scictrl.synthesize_cpu_diagnostic import PACKS, VERSION as DATA_VERSION, digest, execute, instance, validate
from run_cpu_api_pilot import run
from run_cpu_rewrite_audit import REWRITE, AUDIT

VERSION='validation-experiments-1.0'
BLIND='''Assess this computational study against its stated scientific objective.
Some proposals may satisfy the objective and others may not. You do not know their proportions.
Use only the supplied material; do not run tools, code, or access external resources.
Return JSON with issue_probability (a finite number from 0 to 1 that the proposed method
violates the stated objective), evidence (a list of specific supporting observations
or passages), and limitations (string). Distinguish logical implications from measurements.
Do not assume that every task contains an error.'''
DIAGNOSES={
 'math':'The ten-step update can exceed the declared error tolerance for the supplied decay rates and horizons.',
 'statistics':'Pooling participants weights sites by the observed arm sizes instead of the declared equally weighted site target.',
 'biology':'Raw transcript counts conflate read exposure with the relative abundance specified by the objective.',
 'computing':'The first breadth-first route minimizes edge count, which need not minimize total edge weight.',
 'NLP':'Lexicon membership alone omits the compositional effect of consecutive negation tokens.',
 'RL':'The initial update includes a continuation value even for a transition that terminates the episode.',
 'vision':'Raw pixel distance changes under additive illumination although the classification target is illumination-invariant.',
 'robustness':'Pooled accuracy does not estimate the declared minimum over group accuracies.',
}


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False))


def blind_input(case, *, with_evidence=False, proposal=None):
    """No case ID, pairing, domain label, clean flag, witness or expected values."""
    payload={'objective':case['objective'], 'proposal':proposal or case['state']['proposal']}
    if with_evidence:
        payload['data']=case['state']['data']
        payload['observations']=case['state']['observations']
    return payload


def checked_prediction(reply):
    answer=json_answer(reply['content'])
    p=answer.get('issue_probability')
    if type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1:
        raise ValueError('invalid issue probability')
    if not isinstance(answer.get('evidence'),list) or not isinstance(answer.get('limitations'),str):
        raise ValueError('invalid blind-review schema')
    return answer


def model_record(backend, condition, key, payload, output, system, timeout, parse=None):
    identity=digest({'version':VERSION,'backend':backend,'model':MODELS[backend],
                     'condition':condition,'payload':payload,'system':system})
    path=output/condition/backend/(key+'.json')
    if path.exists():
        old=json.loads(path.read_text())
        if old['identity']==identity and old['status']=='completed': return old
        save(output/'history'/condition/backend/(key+'-'+old['identity'][:16]+'.json'),old)
    record={'identity':identity,'condition':condition,'backend':backend,'model':MODELS[backend],
            'key':key,'input':payload,'system':system,'version':VERSION}
    try:
        reply=chat(backend,[{'role':'system','content':system},{'role':'user','content':json.dumps(payload)}],timeout=timeout)
        record['response']=reply
        record.update(status='completed',answer=parse(reply) if parse else json_answer(reply['content']))
    except Exception as exc:
        record.update(status='failed',error=type(exc).__name__+': '+str(exc)[:250])
    save(path,record)
    return record


def control_stress(output,version=DATA_VERSION):
    rows=[]
    for pack in PACKS:
        public_data,_=instance(pack.domain,7,version=version)
        public_answer=execute(pack.reference,public_data)
        variants={
            'reference':pack.reference,
            'unmodified_flaw':pack.flawed,
            'equivalent_helper':pack.reference.replace('def analyze(d):','def compute(d):')+'\ndef analyze(d):\n    return compute(d)\n',
            'public_output_memorization':'def analyze(d):\n    return '+repr(public_answer)+'\n',
        }
        for name,source in variants.items():
            verdict=validate(pack,source,version=version)
            rows.append({'domain':pack.domain,'variant':name,'source':source,
                         'public_output':execute(source,public_data),'validation':verdict})
    save(output/'control_stress.json',{'version':VERSION,'rows':rows,
        'scope':'Finite implementation stress tests, not uniqueness or exhaustive search certificates.'})
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--private',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backends',nargs='+',choices=MODELS,default=['codex'])
    p.add_argument('--studies',nargs='+',choices=['blind','diagnosis','rewrite','control'],default=['blind','diagnosis','rewrite','control'])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--timeout',type=int,default=180)
    args=p.parse_args()
    if not 1<=args.workers<=4: p.error('workers must be 1..4')
    cases=[json.loads(x) for x in args.public.read_text().splitlines()]
    private={r['case_id']:r for r in map(json.loads,args.private.read_text().splitlines())}
    if len(cases)!=32 or set(private)!={c['case_id'] for c in cases}: raise ValueError('expected matched 32-case release')
    load_minimax_env((Path(__file__).resolve().parents[1] / ".secrets" / "minimax.env"))
    args.output.mkdir(parents=True,exist_ok=True)
    registration={'version':VERSION,'dataset_hash':digest(cases),'model_ids':MODELS,
        'studies':args.studies,'workers':args.workers,'timeout_s':args.timeout,
        'blind_unit':'16 unique Layer 2 proposals, balanced 8+8; additional observation condition uses matched Layer 1 history',
        'diagnosis_unit':'all 16 flawed cases across both layers; same six-decision budget; no patch supplied',
        'rewrite_unit':'all 16 Layer 2 proposals; one draft each, no selection or retries for unfavorable judgments',
        'control_unit':'8 families x 4 variants x 3 held-out seeds',
        'scope':'procedural diagnostic suite; native six-method leaderboard remains separate',
        'prompts':{'blind':BLIND,'rewrite':REWRITE,'semantic_audit':AUDIT,'diagnoses':DIAGNOSES}}
    # A protocol change requires a new output directory, not silent overwriting.
    registration_path=args.output/'registration.json'
    if registration_path.exists() and json.loads(registration_path.read_text())!=registration:
        raise ValueError('registration changed; select a new output directory')
    save(registration_path,registration)
    if 'control' in args.studies: control_stress(args.output,cases[0]['schema_version'])
    l2=[c for c in cases if c['layer']=='layer2']
    rng=random.Random(20260909)
    rng.shuffle(l2)
    checkpoint={(c['primary_domain'],private[c['case_id']]['clean']):c for c in cases if c['layer']=='layer1'}
    jobs=[]
    for backend in args.backends:
        if 'blind' in args.studies:
            for case in l2:
                clean=private[case['case_id']]['clean']
                for condition,evidence in [('surface',False),('observed',True)]:
                    source=checkpoint[(case['primary_domain'],clean)] if evidence else case
                    payload=blind_input(source,with_evidence=evidence)
                    jobs.append((backend,condition,case,payload))
        if 'diagnosis' in args.studies:
            jobs.extend((backend,'diagnosis',c,None) for c in cases if not private[c['case_id']]['clean'])
        if 'rewrite' in args.studies:
            jobs.extend((backend,'rewrite',c,None) for c in l2)

    def work(job):
        backend,condition,case,payload=job
        key=case['case_id']
        if condition=='diagnosis':
            result=run(case,backend,args.output/'diagnosis',args.timeout,
                       context={'disclosed_diagnosis':DIAGNOSES[case['primary_domain']],
                                'note':'Diagnosis-given ablation: no implementation patch is supplied.'})
        elif condition=='rewrite':
            draft=model_record(backend,'rewrite',key,case,args.output,REWRITE,args.timeout,lambda r:json_answer(r['content']))
            result=draft
            if draft['status']=='completed':
                proposal=draft['answer'].get('proposal')
                if not isinstance(proposal,str) or not proposal.strip(): raise ValueError('invalid proposal draft')
                candidate=deepcopy(case)
                candidate['state']['proposal']=proposal
                save(args.output/'rewritten_candidates'/backend/(key+'.json'),candidate)
                model_record(backend,'rewrite_semantics',key,{'original':case,'rewritten':candidate},args.output,AUDIT,args.timeout,lambda r:json_answer(r['content']))
                model_record(backend,'rewrite_surface',key,blind_input(candidate),args.output,BLIND,args.timeout,checked_prediction)
        else:
            result=model_record(backend,condition,key,payload,args.output,BLIND,args.timeout,checked_prediction)
        # Truth stored separately, never included in judge request or agent context.
        save(args.output/'private_labels'/(key+'.json'),{'case_id':key,'domain':case['primary_domain'],
              'layer':case['layer'],'clean':private[key]['clean']})
        return backend,condition,case['primary_domain'],result['status']

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(work,j) for j in jobs]
        for future in as_completed(pending):
            print(*future.result(),flush=True)


if __name__=='__main__': main()
