"""Public CPU pilot input envelope. Never access or translate the private oracle."""
import json
from pathlib import Path
from scictrl.public_contract import VERSION

FIELDS={'schema_version','case_id','primary_domain','layer','objective','state','contract','budget'}


def render_cpu_state(case: dict, workspace: Path):
    if set(case)!=FIELDS or case['schema_version'] not in {VERSION,'cpu-pilot-1.0','cpu-pilot-1.1','cpu-pilot-1.2'}:
        raise ValueError('CPU public schema mismatch or private field leakage')
    if set(case['state'])!={'proposal','data','analysis_source','observations'}:
        raise ValueError('public state schema mismatch')
    forbidden={'oracle','gold_state','reference_source','flawed_source','expected','clean','k','target_slot','revision_role'}
    def inspect(value):
        if isinstance(value,dict):
            if set(value)&forbidden: raise ValueError('private annotation in nested public input')
            for child in value.values(): inspect(child)
        elif isinstance(value,list):
            for child in value: inspect(child)
    inspect(case)
    if case['layer'] not in {'layer1','layer2'}:
        raise ValueError('unsupported layer')
    if any(type(v) is not int or v<=0 for v in case['budget'].values()):
        raise ValueError('invalid execution budget')
    if case['layer']=='layer2' and (case['state']['observations'] or case['state']['analysis_source'] is not None):
        raise ValueError('idea start cannot contain invented checkpoint artifacts')
    workspace.mkdir(parents=True,exist_ok=True)
    (workspace/'public_state.json').write_text(json.dumps(case,indent=2))
    (workspace/'data.json').write_text(json.dumps(case['state']['data'],indent=2))
    brief=(case['objective']+'\n\nInitial study proposal: '+case['state']['proposal']+
           '\n\nPublic commitments:\n'+json.dumps(case['contract'],indent=2)+
           '\n\nResources and measured checkpoint, if any:\n'+json.dumps(case['state'],indent=2))
    (workspace/'task.md').write_text(brief)
    if case['state']['analysis_source'] is not None:
        (workspace/'analysis.py').write_text(case['state']['analysis_source'])
    return brief
