"""Generate manuscript evidence from recorded runs, never from intended outcomes."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import yaml

from scictrl.synthesize_cpu_diagnostic import PACKS, digest
from run_cpu_api_pilot import SYSTEM
from run_cpu_rewrite_audit import REWRITE, AUDIT
from run_validation_experiments import BLIND, DIAGNOSES


def esc(value):
    trans={'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&',
           '#':r'\#','$':r'\$','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(trans.get(c,c) for c in str(value))


def listing(value):
    text=value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=True)
    if r'\end{lstlisting}' in text: raise ValueError('unsafe TeX listing input')
    return '\\begin{lstlisting}\n'+text+'\n\\end{lstlisting}\n'


def table(caption, headers, rows, label=None):
    return '\n'.join([r'\par',r'\begin{table}[htbp]\centering\small',r'\caption{'+caption+'}',
        (r'\label{'+label+'}') if label else '',
        r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',
        ' & '.join(headers)+r'\\\midrule',*[' & '.join(map(str,r))+r'\\' for r in rows],
        r'\bottomrule\end{tabular}\end{table}',r'\par'])+'\n'


def read_rows(folder):
    return [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))] if folder.exists() else []


def frac(n,d):return f'{n}/{d}' if d else r'\pending'


def auc(rows):
    pos=[r['probability'] for r in rows if r['label']]
    neg=[r['probability'] for r in rows if not r['label']]
    if not pos or not neg:return None
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))


def blind_metrics(records,labels):
    rows=[]
    for r in records:
        if r['status']!='completed':continue
        p=r['answer']['issue_probability']
        label=not labels[r['key']]['clean']
        rows.append({'id':r['key'],'domain':labels[r['key']]['domain'],
                     'probability':p,'label':label,'correct':(p>=.5)==label})
    pos=[r for r in rows if r['label']]
    neg=[r for r in rows if not r['label']]
    ba=.5*(sum(r['correct'] for r in pos)/len(pos)+sum(r['correct'] for r in neg)/len(neg)) if pos and neg else None
    return {'records':len(records),'completed':len(rows),'flawed_correct':sum(r['correct'] for r in pos),
            'clean_correct':sum(r['correct'] for r in neg),'n_flawed':len(pos),'n_clean':len(neg),
            'balanced_accuracy':ba,'auc':auc(rows),'rows':rows}


def native_prompts(root):
    """Static extraction: no importing third-party agents or their tools."""
    out=[]
    def constants(path,names):
        tree=ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node,ast.Assign):
                for t in node.targets:
                    if isinstance(t,ast.Name) and t.id in names:
                        out.append((path,t.id,ast.literal_eval(node.value)))
    path=root/'Method/AutoResearchClaw/researchclaw/prompts/ml.py'
    tree=ast.parse(path.read_text())
    bank=next(n.value for n in tree.body if isinstance(n,ast.AnnAssign) and getattr(n.target,'id','')=='_DEFAULT_STAGES')
    for key,value in zip(bank.keys,bank.values):
        if ast.literal_eval(key) in {'result_analysis','research_decision'}:
            out.append((path,ast.literal_eval(key),ast.literal_eval(value)))
    path=root/'Method/SAGE/prompts.default.yaml'
    bank=yaml.safe_load(path.read_text())['stages']
    for name in ['result_analysis','research_decision']:
        out.append((path,name,bank[name]))
    constants(root/'Method/AI-Scientist-v2/ai_scientist/treesearch/log_summarization.py',{'report_summarizer_sys_msg','stage_aggregate_prompt'})
    constants(root/'Method/EvoScientist/EvoScientist/prompts.py',{'EVOSCIENTIST_IDENTITY','_EXPERIMENT_WORKFLOW_PREAMBLE'})
    constants(root/'Method/InternAgent-1.5/internagent/prompts.py',{'CODER_PROMPT_MCTS_DRAFT','NEXT_EXPERIMENT_PROMPT'})
    path=root/'Method/AI-Researcher/research_agent/inno/agents/inno_agent/exp_analyser.py'
    tree=ast.parse(path.read_text())
    func=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='instructions')
    out.append((path,'experiment analyser instructions',ast.literal_eval(func.body[0].value)))
    return out


def build(root,paper,release,baseline,suite):
    public=[json.loads(x) for x in (release/'public.jsonl').read_text().splitlines()]
    private={r['case_id']:r for r in map(json.loads,(release/'private.jsonl').read_text().splitlines())}
    output=paper/'generated_appendices'
    output.mkdir(exist_ok=True)
    labels={r['case_id']:r for r in read_rows(suite/'private_labels')}
    raw=read_rows(baseline/'codex')
    diagnostic=read_rows(suite/'diagnosis/codex')
    summary={'release':release.name,'baseline_records':len(raw),'diagnosis_records':len(diagnostic),
             'native_main_complete':False,'source_family_count':8,'case_count':len(public)}

    # B: complete real public records, plus explicitly separated private explanations.
    cases=[]
    for pack in PACKS:
        case=next(c for c in public if c['primary_domain']==pack.domain and c['layer']=='layer1' and not private[c['case_id']]['clean'])
        idea=next(c for c in public if c['primary_domain']==pack.domain and c['layer']=='layer2' and not private[c['case_id']]['clean'])
        cases += [r'\subsubsection{'+esc(pack.domain)+': complete checkpoint example}',
          'The following is the complete public JSON record, including the actual input data and measured initial output. '
          'The subsequent construction explanation and witness are private evaluator material, not agent input.',listing(case),
          r'\paragraph{Corresponding idea start.} The matched idea-start identifier is \texttt{'+esc(idea['case_id'])+'}. '
          r'It retains the same objective, proposal, resources and contract, sets \texttt{analysis\_source} to null and \texttt{observations} to an empty list, and uses layer2. '
          'Thus the agent must generate its own implementation and observations; the constructor does not invent a prior failed experiment.',
          r'\paragraph{Private construction rationale.} '+esc(DIAGNOSES[pack.domain])+' '
          'The annotated revision role is '+r'\texttt{'+esc(pack.revision_role)+'}. '
          'An implementation that meets the same obligation is accepted even if it differs from the witness below. '
          'This witness does not establish a unique permissible algorithm.',listing(pack.reference)]
    (output/'domain_cases.tex').write_text('\n'.join(cases))
    prompts=[]
    for name,value in [('API integration controller',SYSTEM),('Whole-proposal rewrite',REWRITE),
                        ('Semantic-preservation review',AUDIT),('Blind validity review',BLIND),('Disclosed diagnoses',DIAGNOSES)]:
        prompts.extend([r'\subsubsection{'+name+'}',listing(value)])
    (output/'experiment_prompts.tex').write_text('\n'.join(prompts))

    native=[]
    manifest=[]
    for path,name,value in native_prompts(root):
        text=value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=True)
        # Escape only typography unsupported by pdfLaTeX; wording is retained in the JSON archive.
        display=text.replace('—','--').replace('–','-').replace('≥','>=').replace('≤','<=').replace('→','->').replace('•','-').replace('’',"'").replace('“','"').replace('”','"')
        relative=str(path.relative_to(root))
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        native += [r'\subsubsection{'+esc(path.parts[path.parts.index('Method')+1])+': '+esc(name)+'}',
            r'Pinned source: \path{'+relative+'}. This is a selected complete native prompt block, '
            'not a claim that the corresponding full controller was executed in this study.',listing(display)]
        manifest.append({'source':relative,'block':name,'file_sha256':sha,'text':value})
    (output/'native_prompts.tex').write_text('\n'.join(native))
    (output/'native_prompt_archive.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))

    # E and main text: measured coverage and numerical/verification diagnostics only.
    group_rows=[]
    for layer in ['layer1','layer2']:
        for clean in [False,True]:
            rows=[r for r in raw if r['layer']==layer and private[r['case_id']]['clean']==clean]
            done=[r for r in rows if r['status']=='completed']
            calls=[len(r['trace']) for r in rows]
            group_rows.append([layer,'unchanged' if clean else 'flawed',frac(len(done),8),
                frac(sum(r['artifact_valid'] for r in done),len(done)),
                frac(sum(r['artifact_valid'] and r['verified_stop'] for r in done),len(done)),
                f'{statistics.mean(calls):.2f}' if calls else r'\pending'])
    completed=[r for r in raw if r['status']=='completed']
    summary['baseline']={'completed':len(completed),'artifact_pass':sum(r['artifact_valid'] for r in completed),
        'artifact_and_verified_stop':sum(r['artifact_valid'] and r['verified_stop'] for r in completed),
        'api_calls':sum(len(r['trace']) for r in raw),'status_counts':dict(Counter(r['status'] for r in raw))}
    measured=table('Measured Codex-5.6 shared-loop results on '+esc(release.name)+'. Numer.: held-out numerical pass; N+V: numerical pass and verified final source. Neither is full scientific CSR.',
        ['Start','State','Complete','Numer.','N+V','Calls/run'],group_rows,'tab:measured-loop')
    (output/'measured_main.tex').write_text(measured)
    domain_rows=[]
    for pack in PACKS:
        by_domain=[r for r in raw if r['domain']==pack.domain]
        values=[]
        for layer in ['layer1','layer2']:
            rows=[r for r in by_domain if r['layer']==layer]
            done=[r for r in rows if r['status']=='completed']
            values.extend([frac(len(done),2),frac(sum(r['artifact_valid'] and r['verified_stop'] for r in done),len(done))])
        domain_rows.append([esc(pack.domain),*values,str(sum(len(r['trace']) for r in by_domain))])
    (output/'domain_results.tex').write_text(table('Current-release domain-level shared-loop results. Each start layer has one flawed and one unchanged state. N+V is numerical correctness plus final-source verification.',
        ['Domain','L1 complete','L1 N+V','L2 complete','L2 N+V','Calls'],domain_rows,'tab:domain-results'))
    failures=[]
    for r in raw:
        if r['status']!='completed' or not r['artifact_valid'] or not r['verified_stop']:
            failures.append({'case_id':r['case_id'],'domain':r['domain'],'layer':r['layer'],'status':r['status'],
                             'artifact_valid':r['artifact_valid'],'verified_stop':r['verified_stop'],'error':r['error']})
    (output/'run_exceptions.tex').write_text('Exact non-passing diagnostic records (an empty list means none under these diagnostic gates, not that all scientific obligations were adjudicated):\n'+listing(failures))
    representative=next((r for r in raw if r['domain']=='statistics' and r['layer']=='layer2' and not private[r['case_id']]['clean']),None)
    if representative:
        trace={'case_id':representative['case_id'],'controller':representative['method'],
            'steps':[{'step':t['step'],'action':t['answer']['action'],
                      'evidence_report':t['answer'].get('evidence_report'),
                      'submitted_source_hash':digest(t['answer'].get('source','')),
                      'observation':t.get('observation')}
                     for t in representative['trace']],
            'final_numerical_valid':representative['artifact_valid'],
            'final_source_verified':representative['verified_stop'],
            'scientific_evidence_adjudication':representative['evidence_adjudication']}
        (output/'representative_trace.tex').write_text(listing(trace))
    else:(output/'representative_trace.tex').write_text('Representative trajectory not yet available.\n')

    # D: blinded classification; truth comes from private labels, not judge self-score.
    audit_rows=[]
    details=[]
    for condition,title in [('surface','Original brief'),('observed','Brief + observations'),('rewrite_surface','Rewritten brief')]:
        metrics=blind_metrics(read_rows(suite/condition/'codex'),labels)
        summary[condition]=metrics
        audit_rows.append([title,frac(metrics['completed'],16),frac(metrics['flawed_correct'],metrics['n_flawed']),
            frac(metrics['clean_correct'],metrics['n_clean']),
            f"{100*metrics['balanced_accuracy']:.1f}" if metrics['balanced_accuracy'] is not None else r'\pending',
            f"{metrics['auc']:.3f}" if metrics['auc'] is not None else r'\pending'])
        for row in metrics['rows']:
            details.append([esc(row['domain']),title,'flawed' if row['label'] else 'unchanged',f"{row['probability']:.3f}",'yes' if row['correct'] else 'no'])
    audit=table('Blind-review measurements. BA is balanced accuracy (percent); AUC uses issue probabilities. Each condition has eight matched clean/flawed pairs, not 16 independent source studies.',
       ['Input','Coverage','Flaw correct','Clean correct','BA','AUC'],audit_rows,'tab:blind-results')
    semantics=read_rows(suite/'rewrite_semantics/codex')
    done=[r for r in semantics if r['status']=='completed']
    preserved=sum(r['answer'].get('semantic_preserved') is True for r in done)
    summary['semantic_review']={'completed':len(done),'model_judged_preserved':preserved}
    audit+='The semantic reviewer marks '+frac(preserved,len(done))+' completed rewrites as preserving the proposed choices. '
    audit+='This is a same-backbone model judgment, not expert-certified semantic fidelity. No favorable-judgment selection is used. '
    audit+='Rewritten-brief accuracy is scored against the intended source label; if a rewrite changes its scientific validity, that value is not a pure hiddenness measurement.\n'
    (output/'blind_results.tex').write_text(audit)
    detail_tex=[r'\begin{longtable}{lllll}',r'\caption{Every blind-review prediction.}\label{tab:all-blind}\\',
        r'\toprule Domain & Input & Truth & Issue probability & Correct\\\midrule',
        *[' & '.join(r)+r'\\' for r in details],r'\bottomrule\end{longtable}']
    (output/'blind_details.tex').write_text('\n'.join(detail_tex))

    # F: paired cases and costs; no causal claim from one stochastic rollout.
    base={r['case_id']:r for r in raw}
    pairs=[]
    for d in diagnostic:
        if d['case_id'] not in base:continue
        b=base[d['case_id']]
        pairs.append({'domain':d['domain'],'layer':d['layer'],'case_id':d['case_id'],
            'base_pass':b['status']=='completed' and b['artifact_valid'],
            'given_pass':d['status']=='completed' and d['artifact_valid'],
            'base_verified':b['status']=='completed' and b['artifact_valid'] and b['verified_stop'],
            'given_verified':d['status']=='completed' and d['artifact_valid'] and d['verified_stop'],
            'base_calls':len(b['trace']),'given_calls':len(d['trace'])})
    pairs.sort(key=lambda r:([p.domain for p in PACKS].index(r['domain']),r['layer']))
    summary['paired_diagnosis']=pairs
    frows=[[esc(r['domain']),r['layer'],int(r['base_pass']),int(r['given_pass']),
            int(r['base_verified']),int(r['given_verified']),f"{r['base_calls']}/{r['given_calls']}"] for r in pairs]
    diagnosis=table('Paired diagnosis-given ablation on flawed cases. B/G: base/given numerical pass; BV/GV add final-source verification; calls are base/given. One rollout per condition, six decisions each.',
        ['Domain','Start','B','G','BV','GV','Calls'],frows,'tab:diagnosis-results')
    diagnosis+='There are '+str(len(pairs))+' observed pairs. Missing runs are not imputed as scientific failures.\n'
    (output/'diagnosis_results.tex').write_text(diagnosis)

    # G: actually execute constructor stress suite elsewhere; only render its outputs here.
    stress_path=suite/'control_stress.json'
    if stress_path.exists():
        stress=json.loads(stress_path.read_text())['rows']
        srows=[]
        variants=['reference','unmodified_flaw','equivalent_helper','public_output_memorization']
        for pack in PACKS:
            values=[]
            for name in variants:
                r=next(r for r in stress if r['domain']==pack.domain and r['variant']==name)
                values.append(frac(sum(c['valid'] for c in r['validation']['checks']),len(r['validation']['checks'])))
            srows.append([esc(pack.domain),*values])
        (output/'control_results.tex').write_text(table('Executable control stress tests: held-out passes out of three seeds. The equivalent helper is a refactoring check, not a different scientific method.',
            ['Domain','Witness','Flaw','Equivalent','Memorize'],srows,'tab:control-stress'))
        summary['control_stress']={name:sum(r['validation']['valid'] for r in stress if r['variant']==name) for name in variants}
    else:(output/'control_results.tex').write_text('Control stress runs are not yet complete.\n')

    summary['diagnostic_suite_complete']=(len(completed)==32 and len(pairs)==16 and
        all(d['status']=='completed' for d in diagnostic) and
        all(summary[c]['completed']==16 for c in ['surface','observed','rewrite_surface']) and
        summary['semantic_review']['completed']==16 and stress_path.exists())

    # Admission is not inferred from artifact passes or a favorable single-judge review.
    admission=[]
    surface={r['id']:r for r in summary['surface']['rows']}
    for case in public:
        review=surface.get(case['case_id'])
        admission.append({'case_id':case['case_id'],'domain':case['primary_domain'],
            'release':release.name,'diagnostic_artifact_only':True,'native_main_admitted':False,
            'expert_scientific_review':'pending','evidence_dependence':'not_established',
            'surface_issue_probability':review['probability'] if review else None,
            'surface_warning':bool(review and review['label'] and review['probability']>=.5),
            'reason':'Numerical repair feasibility is not sufficient for long-horizon research-task admission.'})
    (suite/'admission_review.json').write_text(json.dumps(admission,indent=2))
    (output/'actual_results.json').write_text(json.dumps(summary,indent=2))
    provenance={'dataset_sha256':digest(public),'baseline_dir':str(baseline),'suite_dir':str(suite),
        'source_record_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [baseline/'codex',suite/'surface/codex',suite/'observed/codex',suite/'rewrite_surface/codex',suite/'diagnosis/codex']
            for p in folder.glob('*.json')}}
    (output/'report_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in {'surface','observed','rewrite_surface','paired_diagnosis'}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--paper',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--suite',type=Path,required=True)
    a=p.parse_args()
    build(a.root.resolve(),a.paper.resolve(),a.release.resolve(),a.baseline.resolve(),a.suite.resolve())
