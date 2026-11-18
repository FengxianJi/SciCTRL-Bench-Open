"""Explicit two-backend API transport. No fallback or local model inference."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

MODELS={'minimax':'MiniMax-M3','codex':'gpt-5.6-sol'}


def load_minimax_env(path: Path):
    if os.environ.get('MINIMAX_API_KEY'):
        return True
    if not path.is_file():
        return False
    if path.stat().st_mode & 0o077:
        raise PermissionError('MiniMax credential file must have mode 600')
    for line in path.read_text().splitlines():
        match=re.fullmatch(r'(?:export\s+)?MINIMAX_API_KEY\s*=\s*(.+)',line.strip())
        if match:
            os.environ['MINIMAX_API_KEY']=match[1].strip().strip('\"\'')
            return True
    return False


def chat(backend, messages, *, timeout=180, max_tokens=3000, reasoning_effort=None, output_schema=None):
    start=time.monotonic()
    model=MODELS[backend]
    if backend=='minimax':
        if not os.environ.get('MINIMAX_API_KEY'):
            raise RuntimeError('MINIMAX_API_KEY unavailable; no fallback attempted')
        payload={'model':model,'messages':messages,'max_tokens':max_tokens,
                 'temperature':1.0,'stream':False,'thinking':{'type':'disabled'}}
        request=urllib.request.Request('https://api.minimaxi.com/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json',
            'Authorization':'Bearer '+os.environ['MINIMAX_API_KEY']})
        try:
            with urllib.request.urlopen(request,timeout=timeout) as response:
                wire=json.load(response)
        except urllib.error.HTTPError as error:
            # Do not echo remote headers, credentials, or a possibly sensitive body.
            raise RuntimeError(f'MiniMax HTTP {error.code}; no fallback attempted') from None
        returned=wire.get('model','')
        if returned and returned.lower()!=model.lower():
            raise RuntimeError('Endpoint returned a different model; run rejected')
        content=wire['choices'][0]['message']['content']
        usage=wire.get('usage',{})
    else:
        with tempfile.TemporaryDirectory(prefix='scictrl-api-only-') as name:
            folder=Path(name)
            output=folder/'answer.txt'
            prompt='This is an API transport call. Do not invoke tools, inspect files, or access the network. Return only the requested answer.\n'+json.dumps(messages)
            command=['codex','exec','--ephemeral','--skip-git-repo-check','--sandbox','read-only',
                     '--model',model,'--cd',name,'--output-last-message',str(output),'-']
            if reasoning_effort is not None:
                if reasoning_effort not in {'low','medium','high'}:raise ValueError('unsupported reasoning effort')
                command[-1:-1]=['--config',f'model_reasoning_effort="{reasoning_effort}"']
            if output_schema is not None:
                schema_file=folder/'response_schema.json'
                schema_file.write_text(json.dumps(output_schema))
                command[-1:-1]=['--output-schema',str(schema_file)]
            result=subprocess.run(command,input=prompt,text=True,capture_output=True,timeout=timeout)
            if result.returncode or not output.is_file():
                raise RuntimeError(f'Codex transport failed (exit {result.returncode}); no fallback attempted')
            content=output.read_text()
            match=re.search(r'tokens used\s+([0-9,]+)',result.stderr)
            usage={'total_tokens':int(match[1].replace(',',''))} if match else {}
            returned=model
    return {'content':content,'requested_model':model,'returned_model':returned,
            'model_identity_evidence':('explicit CLI model flag; server response ID not exposed' if backend=='codex' else 'response.model'),
            'usage':usage,'elapsed_s':round(time.monotonic()-start,3),
            'transport':'codex_cli_authenticated_api' if backend=='codex' else 'minimax_cn_chat_completions',
            'thinking':'disabled' if backend=='minimax' else 'provider_configured',
            'max_tokens_enforced':backend=='minimax'}


def json_answer(text):
    match=re.fullmatch(r'\s*```(?:json)?\s*(.*?)\s*```\s*',text,re.S)
    value=json.loads(match[1] if match else text)
    if not isinstance(value,dict): raise ValueError('JSON object required')
    return value
