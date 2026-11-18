"""Eight executable CPU diagnostic families, separate from legacy structural cases.

These are procedural integration tasks, NOT admitted real-world research studies.
Scientific outputs come from real arithmetic execution, never canned observations.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


VERSION = 'cpu-pilot-1.3'
RELEASE_DIR = 'eight_domains_v4'

REFERENCE_PROPOSALS = {
    'math': 'Use the exponential solution to study the decay dynamics.',
    'statistics': 'Summarize treatment response within sites before averaging site contrasts.',
    'biology': 'Use the ratio of mean exposure-normalized abundances to quantify expression response.',
    'computing': 'Search routes by cumulative weight and retain the minimum-cost route reaching the target.',
    'NLP': 'Use sentiment lexicon membership together with local negation as a compact text representation.',
    'RL': 'Use continuation-aware bootstrapped updates for transitions in the replay record.',
    'vision': 'Compare illumination-centered pixel distances against the reference prototypes.',
    'robustness': 'Calculate group-specific accuracies and use their minimum to summarize reliability.',
}


@dataclass(frozen=True)
class Pack:
    domain: str
    objective: str
    proposal: str
    revision_role: str
    flawed: str
    reference: str


PACKS = (
    Pack('math', 'Predict x(T) for dx/dt=-rate*x, x(0)=1. Absolute error must be below 0.0001 for each supplied experiment.',
         'Use ten equal forward updates to study the decay dynamics.', 'numerical_method',
         'def analyze(d):\n    return [(1-r*t/10)**10 for r,t in d["experiments"]]\n',
         'def analyze(d):\n    return [math.exp(-r*t) for r,t in d["experiments"]]\n'),
    Pack('statistics', 'Estimate the treatment-minus-control mean outcome in a target population that assigns equal weight to every site. Each site contains both arms; report the contrast as a scalar.',
         'Summarize treatment response by pooling all observed participants in each arm.', 'estimand_aggregation',
         'def analyze(d):\n    a=[r["y"] for r in d["rows"] if r["arm"]==1]\n    b=[r["y"] for r in d["rows"] if r["arm"]==0]\n    return sum(a)/len(a)-sum(b)/len(b)\n',
         'def analyze(d):\n    effects=[]\n    for s in sorted(set(r["site"] for r in d["rows"])):\n        a=[r["y"] for r in d["rows"] if r["site"]==s and r["arm"]==1]\n        b=[r["y"] for r in d["rows"] if r["site"]==s and r["arm"]==0]\n        effects.append(sum(a)/len(a)-sum(b)/len(b))\n    return sum(effects)/len(effects)\n'),
    Pack('biology', 'For the synthetic single-transcript study, estimate log2 of the treated/control ratio of mean relative abundance per sample. Library is the total read exposure of each sample.',
         'Use the ratio of mean transcript counts to quantify expression response.', 'normalization',
         'def analyze(d):\n    a=[r["count"] for r in d["rows"] if r["arm"]==1]\n    b=[r["count"] for r in d["rows"] if r["arm"]==0]\n    return math.log2((sum(a)/len(a))/(sum(b)/len(b)))\n',
         'def analyze(d):\n    a=[r["count"]/r["library"] for r in d["rows"] if r["arm"]==1]\n    b=[r["count"]/r["library"] for r in d["rows"] if r["arm"]==0]\n    return math.log2((sum(a)/len(a))/(sum(b)/len(b)))\n'),
    Pack('computing', 'Return minimum total edge cost from node 0 to node n-1 in a directed graph with positive weights. All instances have a path.',
         'Search routes by breadth-first traversal and retain the first route reaching the target.', 'algorithm_objective',
         'def analyze(d):\n    queue=[(0,0)]\n    seen={0}\n    while queue:\n        node,cost=queue.pop(0)\n        if node==d["n"]-1: return cost\n        for a,b,w in d["edges"]:\n            if a==node and b not in seen:\n                seen.add(b)\n                queue.append((b,cost+w))\n',
         'def analyze(d):\n    dist=[1e30]*d["n"]\n    dist[0]=0\n    for _ in range(d["n"]-1):\n        for a,b,w in d["edges"]:\n            dist[b]=min(dist[b],dist[a]+w)\n    return dist[-1]\n'),
    Pack('NLP', 'Predict polarity (+1 or -1) for each synthetic text. good/great encode positive and bad/poor negative sentiment. Negation composes: every token in the consecutive run of not tokens before the sentiment word flips polarity, so two not tokens cancel. Other tokens are neutral.',
         'Use sentiment lexicon membership as a compact text representation.', 'representation',
         'def analyze(d):\n    return [1 if any(w in t.split() for w in ["good","great"]) else -1 for t in d["texts"]]\n',
         'def analyze(d):\n    out=[]\n    for t in d["texts"]:\n        words=t.split()\n        v=1 if words[-1] in ["good","great"] else -1\n        for w in reversed(words[:-1]):\n            if w!="not": break\n            v=-v\n        out.append(v)\n    return out\n'),
    Pack('RL', 'Compute the one-step discounted episodic return target for every transition: reward plus future value when the episode continues, with no future episode contribution after termination.',
         'Use the same bootstrapped update for every transition in the replay record.', 'terminal_semantics',
         'def analyze(d):\n    return [r["reward"]+d["gamma"]*r["next_value"] for r in d["rows"]]\n',
         'def analyze(d):\n    return [r["reward"]+(0 if r["done"] else d["gamma"]*r["next_value"]) for r in d["rows"]]\n'),
    Pack('vision', 'Classify each two-by-two image (flattened in row-major order) as its nearest supplied prototype up to arbitrary additive illumination. Report prototype indices for all images.',
         'Compare raw pixel distances against the reference prototypes.', 'preprocessing',
         'def analyze(d):\n    return [min(range(len(d["prototypes"])),key=lambda i:sum((a-b)**2 for a,b in zip(x,d["prototypes"][i]))) for x in d["images"]]\n',
         'def analyze(d):\n    p=[[v-sum(x)/len(x) for v in x] for x in d["prototypes"]]\n    xs=[[v-sum(x)/len(x) for v in x] for x in d["images"]]\n    return [min(range(len(p)),key=lambda i:sum((a-b)**2 for a,b in zip(x,p[i]))) for x in xs]\n'),
    Pack('robustness', 'Report the minimum accuracy across every declared population group. Each pair is (correct predictions, number of examples); all groups must be represented.',
         'Pool correct predictions and sample counts to summarize reliability.', 'evaluation',
         'def analyze(d):\n    return sum(x[0] for x in d["groups"])/sum(x[1] for x in d["groups"])\n',
         'def analyze(d):\n    return min(a/b for a,b in d["groups"])\n'),
)


def instance(domain: str, seed: int, *, version=VERSION):
    """Resource generator and independent mathematical target (not model output)."""
    rng = random.Random(seed)
    if domain == 'math':
        exps = [[rng.uniform(.5, 2), rng.uniform(1, 4)] for _ in range(8)]
        return {'experiments': exps}, [math.exp(-r*t) for r,t in exps]
    if domain == 'statistics':
        effect = rng.uniform(.3, 2)
        rows = []
        for site in range(3):
            for arm in [0, 1]:
                n = (4 if arm == 0 else 22) if site == 0 else (22 if arm == 0 else 4)
                for i in range(n):
                    rows.append({'site': site, 'arm': arm, 'y': 5*site + effect*arm + (i-(n-1)/2)*.01})
        rng.shuffle(rows)
        return {'rows': rows}, effect
    if domain == 'biology':
        ratio = rng.uniform(1.4, 3)
        rows = []
        for arm in [0, 1]:
            for _ in range(10):
                library = rng.randint(200, 400) * (8 if arm == 0 else 1)
                rows.append({'arm': arm, 'library': library, 'count': library * .01 * (ratio if arm else 1)})
        return {'rows': rows}, math.log2(ratio)
    if domain == 'computing':
        n = rng.randint(5, 10)
        weights = [rng.randint(1, 4) for _ in range(n-1)]
        return {'n': n, 'edges': [[0,n-1,100]] + [[i,i+1,w] for i,w in enumerate(weights)]}, sum(weights)
    if domain == 'NLP':
        texts, labels = [], []
        for i in range(12):
            word = rng.choice(['good', 'great', 'bad', 'poor'])
            negations = i % 3
            texts.append('review is ' + 'not '*negations + word)
            labels.append((1 if word in ['good','great'] else -1) * (-1)**negations)
        return {'texts': texts}, labels
    if domain == 'RL':
        gamma = rng.uniform(.8, .99)
        rows = [{'reward': rng.uniform(-1,2), 'next_value': rng.uniform(2,9), 'done': bool(i%2)} for i in range(8)]
        return {'gamma': gamma, 'rows': rows}, [r['reward'] if r['done'] else r['reward']+gamma*r['next_value'] for r in rows]
    if domain == 'vision':
        p = [[1,-1,1,-1], [11,11,9,9]]
        labels = [i%2 for i in range(12)]
        if version == 'cpu-pilot-1.3':
            rng.shuffle(labels)
        elif version not in {'cpu-pilot-1.0','cpu-pilot-1.1','cpu-pilot-1.2'}:
            raise ValueError('unknown data generator version')
        images = [[v + (18 if label == 0 else -18) + rng.uniform(-.05,.05) for v in p[label]] for label in labels]
        return {'prototypes': p, 'images': images}, labels
    if domain == 'robustness':
        a, b = rng.randint(70,95), rng.randint(1,5)
        return {'groups': [[a,100],[b,10]]}, b/10
    raise ValueError(domain)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_source(source: str):
    """Conservative pure-Python subset; no IO/imports/reflection/process control."""
    if not isinstance(source, str) or len(source) > 30000:
        raise ValueError('source size/type rejected')
    tree = ast.parse(source)
    banned = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef)
    safe_attrs = {'exp', 'log2', 'sqrt', 'log', 'isfinite', 'append', 'pop', 'add', 'items', 'values', 'keys', 'get', 'split', 'sort', 'count', 'index'}
    allowed_calls = {'analyze', 'sum', 'len', 'min', 'max', 'range', 'zip', 'enumerate', 'sorted', 'reversed', 'set', 'list', 'dict', 'tuple', 'float', 'int', 'abs', 'any', 'all', 'round', 'bool'}
    allowed_calls.update(n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and not n.name.startswith('_'))
    for node in ast.walk(tree):
        if isinstance(node, banned):
            raise ValueError('IO/imports/global state are unavailable in this pilot')
        if isinstance(node, ast.Name) and (node.id.startswith('_') and node.id != '_'):
            raise ValueError('private/reflection name rejected')
        if isinstance(node, ast.Attribute) and node.attr not in safe_attrs:
            raise ValueError('attribute unavailable in pure-analysis pilot')
        if isinstance(node, ast.Call) and not ((isinstance(node.func, ast.Name) and node.func.id in allowed_calls) or isinstance(node.func, ast.Attribute)):
            raise ValueError('call unavailable in pure-analysis pilot')


WORKER = '''import sys,json,math,resource
resource.setrlimit(resource.RLIMIT_CPU,(5,5))
resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456))
p=json.load(sys.stdin)
allowed={k:getattr(__builtins__,k) for k in ['sum','len','min','max','range','zip','enumerate','sorted','reversed','set','list','dict','tuple','float','int','abs','any','all','round','bool']}
scope={'__builtins__':allowed,'math':math}
exec(compile(p['source'],'analysis.py','exec'),scope)
print(json.dumps(scope['analyze'](p['data']),allow_nan=False))
'''


def execute(source: str, data: dict, timeout: int = 30):
    check_source(source)
    with tempfile.TemporaryDirectory(prefix='scictrl-cpu-') as temp:
        run = subprocess.run([sys.executable, '-I', '-c', WORKER], input=json.dumps({'source': source, 'data': data}),
                             text=True, capture_output=True, timeout=timeout, cwd=temp,
                             env={'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'})
    if run.returncode:
        raise RuntimeError(f'experiment exited {run.returncode}: {run.stderr[-1000:]}')
    if len(run.stdout) > 100000:
        raise ValueError('output too large')
    return json.loads(run.stdout)


def close(actual, expected, tolerance=1e-4):
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual)==len(expected) and all(close(a,b,tolerance) for a,b in zip(actual,expected))
    return type(actual) in (int,float) and math.isfinite(actual) and abs(actual-expected) < tolerance


def validate(pack: Pack, source: str, seeds=(101,202,303), *, version=VERSION):
    rows = []
    for seed in seeds:
        data, expected = instance(pack.domain, seed,version=version)
        try:
            actual = execute(source, data)
            rows.append({'seed': seed, 'valid': close(actual,expected), 'actual': actual,
                         'data_hash': digest(data), 'expected': expected})
        except (ValueError, SyntaxError, RuntimeError, subprocess.TimeoutExpired) as error:
            rows.append({'seed': seed, 'valid': False, 'error': type(error).__name__})
    return {'valid': all(x['valid'] for x in rows), 'checks': rows, 'source_hash': digest(source)}


def public_case(pack: Pack, layer: str, clean: bool, seed=7):
    data, _ = instance(pack.domain, seed)
    source = pack.reference if clean else pack.flawed
    observations = []
    if layer == 'layer1':
        observations = [{'action': 'execute_analysis', 'output': execute(source,data),
                         'source_hash': digest(source), 'data_hash': digest(data)}]
    proposal = REFERENCE_PROPOSALS[pack.domain] if clean else pack.proposal
    return {'schema_version': VERSION, 'case_id': digest([pack.domain,layer,clean,seed,VERSION])[:20],
            'primary_domain': pack.domain, 'layer': layer, 'objective': pack.objective,
            'state': {'proposal': proposal, 'data': data, 'analysis_source': source if layer=='layer1' else None,
                      'observations': observations},
            'contract': {'data_immutable': True, 'objective_immutable': True,
                         'numerical_acceptance': 'Each numeric entry must have absolute error strictly below 0.0001; booleans and non-finite numbers are not accepted.',
                         'output': 'pure Python analyze(d), JSON-compatible return; all requested entries, same order',
                         'return_schema': ({'type':'array','items':{'type':'number'},'order':'input order'}
                                           if pack.domain in {'math','NLP','RL','vision'} else {'type':'number'}),
                         'runtime': 'stdlib math provided; no imports, IO, reflection, GPU or external models; pure analysis subset',
                         'deliverables': ['source','measured_output','evidence_report','stop']},
            'budget': {'model_decisions': 6, 'experiment_timeout_s': 30}}


def generate(root: Path, paper_dir: Path | None = None):
    folder = root / 'ExecutablePacks' / RELEASE_DIR
    reports, public, private = [], [], []
    for pack in PACKS:
        gold = validate(pack,pack.reference)
        flawed = validate(pack,pack.flawed)
        if not gold['valid'] or any(x['valid'] for x in flawed['checks']):
            raise RuntimeError(f'construction contrast failed: {pack.domain}')
        reports.append({'domain': pack.domain, 'reference': gold, 'flawed': flawed})
        for layer in ['layer1','layer2']:
            for clean in [False,True]:
                case = public_case(pack,layer,clean)
                public.append(case)
                private.append({'case_id':case['case_id'], 'domain':pack.domain, 'clean':clean,
                                'k':0 if clean else 1, 'revision_role':pack.revision_role,
                                'reference_source':pack.reference, 'source_hash':digest(case),
                                'admission':'cpu_witness_verified_not_expert_admitted'})
    folder.mkdir(parents=True,exist_ok=True)
    for name, rows in [('public.jsonl',public),('private.jsonl',private)]:
        (folder/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    for layer in ['layer1','layer2']:
        target = root / ('Layer1' if layer=='layer1' else 'Layer2') / RELEASE_DIR
        target.mkdir(parents=True,exist_ok=True)
        (target/'public.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in public if x['layer']==layer))
    summary = {'version':VERSION,'families':len(PACKS),'public_cases':len(public),
               'primary_domains':[p.domain for p in PACKS], 'reference_checks':sum(len(x['reference']['checks']) for x in reports),
               'flawed_checks':sum(len(x['flawed']['checks']) for x in reports),
               'reports':reports,'scope':'procedural CPU diagnostic pilot; not native Agent results'}
    (folder/'construction_report.json').write_text(json.dumps(summary,indent=2))
    if paper_dir:
        (paper_dir/'generated_cpu_results.tex').write_text(
            f'The procedural CPU pilot contains {len(PACKS)} source families and {len(public)} public instances '
            '(two start layers, each with an unchanged and a single-intervention state). '
            f'On three held-out data seeds per family, all {summary["reference_checks"]} reference executions satisfy the numerical obligations, '
            f'and all {summary["flawed_checks"]} corresponding flawed executions violate them. '
            'These are constructor-witness checks, not LLM or native-agent success rates.\n')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark-root',type=Path,required=True)
    parser.add_argument('--paper-dir',type=Path)
    args=parser.parse_args()
    result=generate(args.benchmark_root,args.paper_dir)
    print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))
