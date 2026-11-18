"""InternAgent's native experiment/revision module, not its upstream MAS ideation."""
import json
import os
import subprocess

def run(args,case,workspace,bridge):
    from native_anthropic_bridge import serve_anthropic
    from native_bridge import NativeBudgetExceeded
    from agent_adapters.cpu_state import render_cpu_state
    import internagent.experiments_utils_claude as native
    work=workspace/'workspace'
    brief=render_cpu_state(case,work)
    brief+='\nUse only the supplied task and data. CPU only; no local language models, GPUs, dataset downloads, package installations or publication. '
    brief+='Write pure analyze(d) to analysis.py, a launcher.sh that executes the analysis, result.json and report.md. '
    brief+='Execute the code on supplied data. Do not invent experimental observations. Keep all modifications inside the task directory. '
    brief+='Do not fabricate a baseline score when none was supplied.'
    idea={'name':case['case_id'],'title':case['objective'],'description':brief,'method':case['state']['proposal']}
    count=0
    with serve_anthropic(bridge) as (url,token):
        def cli_run(runner,prompt,cwd=None):
            nonlocal count
            if bridge.count>=bridge.max_calls:raise NativeBudgetExceeded('native model-call budget exhausted')
            env=os.environ.copy()
            env.pop('CLAUDE_CODE_OAUTH_TOKEN',None)
            env.update(ANTHROPIC_BASE_URL=url,ANTHROPIC_API_KEY=token,ANTHROPIC_AUTH_TOKEN=token,
                ANTHROPIC_MODEL=bridge.model,ANTHROPIC_DEFAULT_SONNET_MODEL=bridge.model,
                ANTHROPIC_DEFAULT_HAIKU_MODEL=bridge.model,ANTHROPIC_DEFAULT_OPUS_MODEL=bridge.model,
                CLAUDE_CONFIG_DIR=str(workspace/'claude_runtime'),CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',
                MAX_THINKING_TOKENS='0',CUDA_VISIBLE_DEVICES='')
            command=['claude','--print','--output-format','json','--model',bridge.model,'--setting-sources','',
                '--permission-mode','acceptEdits','--allowedTools','Bash,Read,Write,Edit,Glob,Grep','--max-turns','32',prompt]
            result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,timeout=600)
            (workspace/f'native_cli_{count:02d}.json').write_text(json.dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2))
            count+=1
            if result.returncode:
                if 'NativeBudgetExceeded' in result.stdout or 'NativeBudgetExceeded' in result.stderr:
                    raise NativeBudgetExceeded('native model-call budget exhausted')
                raise RuntimeError(f'native Claude CLI exited {result.returncode}; see scoped CLI log')
            return result.stdout
        native.ClaudeCodeRunner.run=cli_run
        original_run=native.run_experiment
        def observed_run(*a,**k):
            result=original_run(*a,**k)
            with (workspace/'native_experiments.jsonl').open('a') as log:log.write(json.dumps({'returncode':result[0],'feedback':result[1],'traceback':result[2],'message':result[3]},default=str)+'\n')
            return result
        native.run_experiment=observed_run
        success=native.perform_experiments(idea,str(work),model=bridge.model,gpu_ids='',max_runs=2,run_timeout=30)
    return {'native_terminal_reached':bool(success),'native_component':'native experiment/revision/report module; upstream MAS ideation not run',
            'native_cli_invocations':count,'token_count_endpoint':'character-based estimate for CLI context management, not measured token usage'}
