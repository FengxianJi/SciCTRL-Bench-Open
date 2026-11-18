"""Report completed native attempts without turning them into a leaderboard."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Method'))
from evaluate_native_cpu import evaluate

METHODS=['AutoResearchClaw','SAGE','AI-Scientist-v2','EvoScientist','AI-Researcher','InternAgent-1.5']
PRIMARY={'native_cpu_v2':['AutoResearchClaw','SAGE'], 'native_evo_cpu_v1':['EvoScientist']}
PAPER=ROOT/'_ICLR27_SciCTRL_Bench__A_Benchmark__For_End_to_End_AI_Scientists_Controlled_Feedback_Revision_Evaluation'

def update_inline(name, content):
    """Update only the marked report block; never recreate split TeX files."""
    main=PAPER/'iclr2027_conference.tex'
    source=main.read_text()
    begin=f'% BEGIN INLINE: {name}\n'
    end=f'% END INLINE: {name}'
    if source.count(begin)!=1 or source.count(end)!=1:
        raise ValueError(f'Missing or ambiguous inline report markers: {name}')
    prefix,rest=source.split(begin,1)
    _,suffix=rest.split(end,1)
    main.write_text(prefix+begin+content.rstrip('\n')+'\n'+end+suffix)

def artifact_category(check):
    if check.get('numerical_valid'): return 'numerical_pass'
    if check.get('artifact_error')=='required final analysis.py absent': return 'missing_artifact'
    if check.get('artifact_error'): return 'execution_or_contract_rejection'
    if check.get('error'): return 'evaluator_error'
    if 'heldout' in check: return 'numerical_mismatch'
    return 'unresolved'

def aggregate(selected,method,backend):
    return {'method':method,'backend':backend,'attempts':len(selected),
        'distinct_cases':len({r['case_id'] for r in selected}),
        'native_terminal':sum(bool(r.get('native_terminal_reached')) for r in selected),
        'artifact_valid':sum(bool(r['artifact_checks'].get('numerical_valid')) for r in selected),
        'terminal_and_valid':sum(bool(r.get('native_terminal_reached')) and bool(r['artifact_checks'].get('numerical_valid')) for r in selected),
        'api_calls':sum(r['api_calls'] for r in selected),
        'statuses':dict(Counter(r['status'] for r in selected)),
        'adapter_revisions':len({r['adapter_source_hash'] for r in selected if r.get('adapter_source_hash')}),
        'missing_adapter_hash':sum(not r.get('adapter_source_hash') for r in selected)}

def write_paper_audit(summary,primary,metadata):
    table=summary['primary_table']
    total=lambda key:sum(r[key] for r in table)
    finished=all(r['distinct_cases']==32 for r in table)
    tex=['% Generated from native result files; primary batches exclude diagnostic retries.',
        '\\paragraph{Native CPU compatibility batches ('+('ended' if finished else 'partial')+').}',
        f"As of {summary['timestamp_utc'][:10]} (UTC), the two primary batches contain {total('attempts')} ended attempts across AutoResearchClaw, SAGE and EvoScientist, each evaluated on 32 records with both backbones.",
        'These are native-controller compatibility runs, distinct from the shared-loop pilot and from full upstream-paper reproductions.',
        'Each attempt has a 32-request cap and a 1,200-second wall-clock limit; GPU work is excluded.',
        '\\begin{table}[htbp]\\centering\\small',
        '\\caption{Primary native CPU batches, excluding debugging retries. T: native terminal reached; N: final artifact passes three held-out seeds; T$\\cap$N: both. N includes clean controls and artifacts from failed runs. None of these columns is controlled scientific success. M3 denotes MiniMax-M3; Codex denotes gpt-5.6-sol.}',
        '\\label{tab:native-runtime-audit}',
        '\\begin{tabular}{llrrrrr}\\toprule',
        'Method & Backbone & Attempts & T & N & T$\\cap$N & Calls \\\\ \\midrule']
    for r in table:
        tex.append(r['method']+' & '+('M3' if r['backend']=='minimax' else 'Codex')+' & '+' & '.join(str(r[k]) for k in ['attempts','native_terminal','artifact_valid','terminal_and_valid','api_calls'])+' \\\\')
    tex+=['\\midrule Total & & '+' & '.join(str(total(k)) for k in ['attempts','native_terminal','artifact_valid','terminal_and_valid','api_calls'])+' \\\\',
        '\\bottomrule\\end{tabular}\\end{table}',
        f"There are {total('native_terminal')} terminal runs and {total('artifact_valid')} numerically valid artifacts, with only {total('terminal_and_valid')} attempts satisfying both gates.",
        'The request count measures recorded bridge invocations, not successful responses, equal token expenditure, or monetary cost.',
        'Several adapter revisions were deployed while these batches were queued; budget exhaustion and integration failures prevent interpreting the differences as a controlled method ranking.',
        'AI Scientist-v2 and InternAgent-1.5 have separate native experiment-module debugging runs, not full-dataset results; AI-Researcher has no measured model attempt in this audit.',
        'Appendix~\\ref{app:native-audit} reports raw failure states, domain/layer breakdowns and the distinct retry inventory.',
        'The strict scientific-success matrix remains unmeasured: numerical validity alone does not establish evidence-grounded identification, integrity or complete downstream verification.',
        '% Snapshot UTC: '+summary['timestamp_utc']]
    update_inline('native_runtime_audit.tex', ('\n'.join(tex)+'\n').replace('\\begin{table}', '\n\\begin{table}').replace('\\end{table}', '\\end{table}\n'))
    detail=['% Generated from native result files and joined, private condition labels.',
        '\\subsection{Native CPU batch audit}\\label{app:native-audit}',
        'This audit covers \\texttt{native\\_cpu\\_v2} and \\texttt{native\\_evo\\_cpu\\_v1}. Each method--backbone cell has 32 planned records: eight domains, two layers and clean/problem states. Debugging attempts below are excluded from these cells.',
        'AutoResearchClaw and SAGE execute their native pipelines from stage 14 (checkpoint start) or stage 9 (idea start) toward stage 23, with public-state checkpoint mappings. EvoScientist uses its native agent graph with CPU tools. These scoped adapters do not reproduce unrestricted upstream ideation, resource discovery or GPU experiments.',
        'The shared-loop six-decision budget does not apply here. Primary native attempts use 32 bridge requests and a 1,200-second process limit; experiments use CPU execution. MiniMax thinking is disabled and Codex requests low reasoning effort. Retry/serialization requests consume the recorded bridge budget; HTTP transport retries can occur within an invocation.',
        '\\paragraph{Raw terminal states, not cognitive diagnoses.}',
        '\\begin{table}[htbp]\\centering\\small',
        '\\caption{Raw process status counts in the primary batches. Done: completed; Budget: budget exhausted; Process: native process failure; Time: wall timeout; Error: other raised failure. Each row sums to 32.}',
        '\\label{tab:native-status}',
        '\\begin{tabular}{llrrrrr}\\toprule Method & Backbone & Done & Budget & Process & Time & Error \\\\ \\midrule']
    for r in table:
        detail.append(r['method']+' & '+('M3' if r['backend']=='minimax' else 'Codex')+' & '+' & '.join(str(r['statuses'].get(k,0)) for k in ['completed','budget_exhausted','native_process_failure','timeout','failed'])+' \\\\')
    detail+=['\\bottomrule\\end{tabular}\\end{table}',
        'Budget or process failure is not automatically failure to identify or repair a scientific problem. For example, AutoResearchClaw/SAGE can reach the paper gate but be blocked for lack of real metrics, even when a selected analysis artifact passes independent numerical checks. Recorded EvoScientist errors include malformed structured replies and MiniMax HTTP 529 overload. These logs do not justify attributing all failures to scientific reasoning.',
        '\\paragraph{Artifact-validation boundary.}',
        'The post-run checker selects a predetermined final \\texttt{analysis.py} path, never the candidate with the best hidden score. For AutoResearchClaw/SAGE it prefers stage-13 refinement, then stage-10 implementation; for the other methods it uses the workspace artifact. It normalizes AST formatting and removes only an unaliased \\texttt{import math}, because the evaluator supplies that module. No repair is synthesized during evaluation.',
        'A missing expected source, invalid source, or failed numerical check contributes no numerical pass; an absent mapped artifact is not proof that no useful work exists elsewhere in the native repository. Three held-out seeds test final behavior. They do not certify an explanation, an unchanged scientific objective, or a valid complete report.',
        'Numerically valid artifacts can survive a failed run; clean controls can pass without repairing anything. Conversely, controller termination can occur with an invalid or missing mapped artifact. Thus neither count is a repair success rate.',
        'Native execution logging was added during integration. Where available, a stronger check matches final source/data hashes and measured JSON output to independent execution. Missing such instrumentation is not proof that execution never occurred; its non-uniform coverage precludes a cross-method execution-success rate.',
        '\\paragraph{Separating numerical mismatch from interface rejection.}',
        '\\begin{table}[htbp]\\centering\\small',
        '\\caption{Post-run validation disposition, independent of native terminal status. Missing artifacts and evaluator-contract rejections are not established scientific reasoning failures.}',
        '\\label{tab:native-validation-disposition}',
        '\\begin{tabular}{lr}\\toprule Disposition & Attempts \\\\ \\midrule']
    counts=Counter(artifact_category(r['artifact_checks']) for r in primary)
    labels={'numerical_pass':'All held-out seeds pass','numerical_mismatch':'Checked, but held-out numerical mismatch',
        'execution_or_contract_rejection':'Source execution / evaluator-contract rejection','missing_artifact':'Expected analysis artifact absent',
        'evaluator_error':'Evaluator infrastructure error','unresolved':'Unresolved check'}
    for key,label in labels.items():
        if counts[key]: detail.append(f'{label} & {counts[key]} \\\\')
    joint=Counter((bool(r.get('native_terminal_reached')),bool(r['artifact_checks'].get('numerical_valid'))) for r in primary)
    detail+=['\\bottomrule\\end{tabular}\\end{table}',
        f"The two gates disagree on {joint[False,True]+joint[True,False]} attempts: {joint[False,True]} have a numerical pass without native termination, whereas {joint[True,False]} terminate without a numerical pass. Another {joint[False,False]} satisfy neither gate.",
        'The current rejected-source messages concern unavailable calls, IO/import/global-state restrictions, or restricted attribute/reflection access in the pure-analysis evaluator. Such rejection establishes incompatibility with this diagnostic contract, not necessarily an incorrect underlying scientific method. The remaining numerical mismatches are behavioral failures on these tests, not automatically failures of identification.',
        '\\paragraph{Layer and initial-state breakdown.}',
        'Table~\\ref{tab:native-layer} separates clean controls from problem-bearing states at each start layer; Table~\\ref{tab:native-domain} reports the eight domain groups. All denominators include failed and timed-out attempts, rather than conditioning on a returned artifact.',
        '\\begin{table}[htbp]\\centering\\small',
        '\\caption{Numerical artifact passes / ended attempts, not scientific repair rates. C: clean; P: problem-bearing. Each cell has eight records, one per domain.}',
        '\\label{tab:native-layer}',
        '\\begin{tabular}{llrrrr}\\toprule Method & Backbone & L1 C & L1 P & L2 C & L2 P \\\\ \\midrule']
    for r in table:
        selected=[a for a in primary if a['method']==r['method'] and a['backend']==r['backend']]
        cells=[]
        for layer,clean in [('layer1',True),('layer1',False),('layer2',True),('layer2',False)]:
            group=[a for a in selected if metadata[a['case_id']]['layer']==layer and metadata[a['case_id']]['clean']==clean]
            cells.append(str(sum(bool(a['artifact_checks'].get('numerical_valid')) for a in group))+'/'+str(len(group)))
        detail.append(r['method']+' & '+('M3' if r['backend']=='minimax' else 'Codex')+' & '+' & '.join(cells)+' \\\\')
    detail+=['\\bottomrule\\end{tabular}\\end{table}',
        '\\begin{table}[htbp]\\centering\\small',
        '\\caption{Numerical passes / ended attempts by domain. Columns distinguish method and backbone; each cell has four records. These are one-family procedural diagnostics, not independent research projects.}',
        '\\label{tab:native-domain}',
        '\\begin{tabular}{lrrrrrr}\\toprule & \\multicolumn{2}{c}{AutoResearchClaw} & \\multicolumn{2}{c}{SAGE} & \\multicolumn{2}{c}{EvoScientist} \\\\',
        'Domain & M3 & Codex & M3 & Codex & M3 & Codex \\\\ \\midrule']
    for domain in ['math','statistics','biology','computing','NLP','RL','vision','robustness']:
        cells=[]
        for r in table:
            group=[a for a in primary if a['method']==r['method'] and a['backend']==r['backend'] and metadata[a['case_id']]['primary_domain']==domain]
            cells.append(str(sum(bool(a['artifact_checks'].get('numerical_valid')) for a in group))+'/'+str(len(group)))
        detail.append(domain+' & '+' & '.join(cells)+' \\\\')
    detail+=['\\bottomrule\\end{tabular}\\end{table}',
        '\\paragraph{Protocol drift and debugging inventory.}',
        'The machine-readable audit retains each result path, input hashes, available adapter hash and validation record. Adapter hashes vary within primary cells; later attempts also have source snapshots. Corrections to checkpoint mapping, tool serialization and execution logging make this a development rollout rather than a frozen-protocol comparison. New matched runs with pinned adapters and scientifically suitable budgets are needed for a ranking.',
        '\\begin{table}[htbp]\\centering\\small',
        '\\caption{Additional debugging attempts only, excluded from the primary 192 attempts. Repeated cases and 32-/64-request budgets are not independent benchmark trials. T/N have the meanings in Table~\\ref{tab:native-runtime-audit}.}',
        '\\label{tab:native-debug}',
        '\\begin{tabular}{llrrrrr}\\toprule Method & Backbone & Attempts & Cases & T & N & Calls \\\\ \\midrule']
    for method in METHODS:
        for backend in ['minimax','codex']:
            selected=[r for r in summary['runs'] if r['run_root'] not in PRIMARY and r['method']==method and r['backend']==backend]
            if not selected:continue
            r=aggregate(selected,method,backend)
            detail.append(method+' & '+('M3' if backend=='minimax' else 'Codex')+' & '+' & '.join(str(r[k]) for k in ['attempts','distinct_cases','native_terminal','artifact_valid','api_calls'])+' \\\\')
    detail+=['\\bottomrule\\end{tabular}\\end{table}',
        'AI Scientist-v2 debugging invokes its native four-stage experiment manager; none of the audited attempts reaches the required terminal stage. InternAgent invokes its native experiment/revision/report module, not upstream MAS ideation; one MiniMax attempt reaches module completion after 43 requests. A SAGE--MiniMax debugging trajectory reaches its terminal stage after 44 requests, illustrating why a 32-request cap can censor completion.',
        'AI-Researcher has no model-call result in the audited inventory; its recorded entrypoint checks identify import/container dependencies, not a GPU skip or a measured model failure.',
        '\\paragraph{Complete native result-file inventory.}',
        'All immediate native run directories are scanned for task-level result records, rather than selecting successful runs or maintaining a debugging allowlist. The audit also retains zero-request startup failures and separately lists topic-initialization smoke tests.',
        '\\begin{table}[htbp]\\centering\\small',
        '\\caption{Inventory units must not be pooled into benchmark success rates. Task-level attempts, pre-request failures and single-stage smoke tests have different scopes.}',
        '\\label{tab:native-inventory}',
        '\\begin{tabular}{lrr}\\toprule Record class & Records & Recorded calls \\\\ \\midrule',
        f"Primary task-level batches & {len(primary)} & {sum(r['api_calls'] for r in primary)} \\\\",
        f"Other task-level attempts with requests & {len(summary['runs'])-len(primary)} & {sum(r['api_calls'] for r in summary['runs'])-sum(r['api_calls'] for r in primary)} \\\\",
        f"Task-level startup failures, zero requests & {len(summary['startup_runs'])} & 0 \\\\",
        f"Separate topic-stage smoke tests & {len(summary['smoke_runs'])} & {sum(r['api_calls'] for r in summary['smoke_runs'])} \\\\",
        '\\bottomrule\\end{tabular}\\end{table}',
        'The complete scan adds a previously omitted EvoScientist--Codex debugging failure with four recorded requests and a structured-response parsing error; it does not change the primary-batch counts. The two topic-stage smoke records concern AutoResearchClaw and SAGE with Codex, one request each, and assess neither experimentation nor revision. Zero-request failures are retained for infrastructure coverage but do not measure model capability.',
        'No native attempt is promoted here to strict controlled scientific success. Evidence-based identification, integrity, downstream consistency and stopping remain separate obligations; their unmeasured values are not set to zero.',
        '% Snapshot UTC: '+summary['timestamp_utc']]
    update_inline('native_runtime_details.tex', ('\n'.join(detail)+'\n').replace('\\begin{table}', '\n\\begin{table}').replace('\\end{table}', '\\end{table}\n'))

def main():
    rows=[]
    startup=[]
    roots=sorted(p for p in (ROOT/'Benchmark/Results').glob('native*') if p.is_dir())
    for directory in roots:
        name=directory.name
        for path in sorted(directory.glob('*/*/*/native_result.json')):
            result=json.loads(path.read_text())
            result['run_root']=name
            result['result_path']=str(path)
            if not result.get('api_calls'):
                startup.append(result)
                continue
            try:
                result['artifact_checks']=evaluate(path.parent)
                (path.parent/'artifact_evaluation.json').write_text(json.dumps(result['artifact_checks'],indent=2))
            except Exception as exc:result['artifact_checks']={'numerical_valid':False,'error':type(exc).__name__}
            rows.append(result)
    smoke=[]
    for path in sorted((ROOT/'Benchmark/Results/native_cpu_smoke').glob('*/native_smoke.json')):
        smoke.append({**json.loads(path.read_text()),'result_path':str(path)})
    table=[]
    for method in METHODS:
        for backend in ['minimax','codex']:
            selected=[r for r in rows if r['method']==method and r['backend']==backend]
            table.append(aggregate(selected,method,backend))
    public=[json.loads(line) for line in (ROOT/'Benchmark/ExecutablePacks/eight_domains_v4/public.jsonl').read_text().splitlines()]
    private={r['case_id']:r for r in [json.loads(line) for line in (ROOT/'Benchmark/ExecutablePacks/eight_domains_v4/private.jsonl').read_text().splitlines()]}
    metadata={r['case_id']:{'primary_domain':r['primary_domain'],'layer':r['layer'],'clean':private[r['case_id']]['clean']} for r in public}
    primary=[r for r in rows if r['run_root'] in PRIMARY]
    primary_table=[]
    for root,methods in PRIMARY.items():
        for method in methods:
            for backend in ['minimax','codex']:
                selected=[r for r in primary if r['run_root']==root and r['method']==method and r['backend']==backend]
                assert len(selected)==len({r['case_id'] for r in selected}), 'Duplicate primary task'
                assert all(r['case_id'] in metadata for r in selected), 'Unknown primary task'
                primary_table.append(aggregate(selected,method,backend))
    timestamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
    summary={'timestamp_utc':timestamp,'scope':'native compatibility attempts, including diagnostic retries; not scientific success rates',
        'complete_leaderboard':False,'table':table,'runs':rows,'primary_table':primary_table,
        'startup_runs':startup,'smoke_runs':smoke,'scanned_roots':[p.name for p in roots],
        'coverage':{name:len(list((ROOT/'Benchmark/Results'/name).glob('*/*/*/native_result.json'))) for name in ['native_cpu_v2','native_evo_cpu_v1']}}
    target=ROOT/'Benchmark/Results/native_run_status.json'
    target.write_text(json.dumps(summary,indent=2))
    lines=['# 原生实验运行快照',f'更新：{timestamp}','',
        '这里只统计已结束且确实发起过模型请求的运行尝试。调试重试、不同预算和适配版本分开保留，不能当作统一主实验成功率。',
        '原生流程到终点 ≠ 科研任务成功；数值通过 ≠ 有证据的受控修正。GPU 工作未启动。','',
        '| 方法 | 基座 | 已结束尝试 | 不同任务 | 原生到终点 | 产物数值通过 | 模型请求 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['method']} | {r['backend']} | {r['attempts']} | {r['distinct_cases']} | {r['native_terminal']} | {r['artifact_valid']} | {r['api_calls']} |")
    lines+=['','主批次（不含调试重试）：','',
        '| 方法 | 基座 | 结果数 | 原生到终点 | 数值通过 | 两者同时 | 模型请求 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in primary_table:lines.append(f"| {r['method']} | {r['backend']} | {r['attempts']} | {r['native_terminal']} | {r['artifact_valid']} | {r['terminal_and_valid']} | {r['api_calls']} |")
    lines+=['','批次结果覆盖（结束不等于成功）：',
        f"- 全目录扫描：{len(rows)} 个有请求的任务级尝试，另有 {len(startup)} 个零请求启动失败、{len(smoke)} 个单阶段 smoke 测试。",
        f"- AutoResearchClaw / SAGE：{summary['coverage']['native_cpu_v2']} / 128 个结果文件。",
        f"- EvoScientist：{summary['coverage']['native_evo_cpu_v1']} / 64 个结果文件。",
        '- AI Scientist-v2：实际四阶段实验管理器调试；论文 launcher 未运行。',
        '- InternAgent：实际 experiment/revision/report 模块调试；上游 MAS ideation 未运行。',
        '- AI-Researcher：导入及 Docker 依赖仍阻塞，不计为 GPU 跳过或已复现。','',
        '基座只使用 MiniMax-M3 与 gpt-5.6-sol。MiniMax thinking 已关闭，Codex 正式兼容批次为 low。',
        '32 次调用并不足以让所有原生方法到终点：一条 SAGE 调试轨迹实际用了 44 次。',
        'MiniMax key 已实际通过认证；发生过的 529 服务繁忙与 401 认证失败不同。',
        'Claude CLI 的费用估计不是 MiniMax 官方账单，不作为实验成本。',
        '公共数据已重新生成并与现有 v4 文件逐字节一致；24 个标准修正检查通过，24 个错误版本检查失败。','',
        '每个原始目录保留 native_result.json、api_calls、原生文件与独立 artifact_evaluation.json。',
        '更新快照：`Method/AutoResearchClaw/.venv/bin/python Method/summarize_native_runs.py`']
    (ROOT/'Method/NATIVE_RUN_STATUS.md').write_text('\n'.join(lines)+'\n')
    write_paper_audit(summary,primary,metadata)
    print(json.dumps({'timestamp':timestamp,'completed_attempts':len(rows),'coverage':summary['coverage']},indent=2))

if __name__=='__main__':main()
