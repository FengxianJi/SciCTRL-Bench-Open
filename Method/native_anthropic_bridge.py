"""Anthropic wire framing for the native Claude CLI used by InternAgent.

Only protocol translation: native CLI selects and executes tools. Every actual
model request still goes through the selected MiniMax/Codex Bridge.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import secrets
import threading

def text_blocks(value):
    if isinstance(value,str):return value
    return '\n'.join(b.get('text','') for b in value or [] if b.get('type')=='text')

def convert_messages(body):
    messages=[]
    if body.get('system'):messages.append({'role':'system','content':text_blocks(body['system'])})
    for item in body['messages']:
        content=item['content']
        if isinstance(content,str):messages.append({'role':item['role'],'content':content});continue
        calls=[]
        for block in content:
            if block['type']=='tool_use':
                calls.append({'id':block['id'],'type':'function','function':{'name':block['name'],'arguments':json.dumps(block['input'])}})
            elif block['type']=='tool_result':
                messages.append({'role':'tool','tool_call_id':block['tool_use_id'],'content':text_blocks(block.get('content',''))})
            elif block['type'] not in {'text','thinking','redacted_thinking'}:
                raise ValueError('unsupported Anthropic content block: '+block['type'])
        text=text_blocks(content)
        if text or calls:
            message={'role':item['role'],'content':text or None}
            if calls:message['tool_calls']=calls
            messages.append(message)
    return messages

@contextmanager
def serve_anthropic(bridge):
    token=secrets.token_hex(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.headers.get('x-api-key')!=token and self.headers.get('Authorization')!='Bearer '+token:
                self.send_error(401);return
            body=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
            if '/count_tokens' in self.path:
                data=json.dumps({'input_tokens':max(1,len(json.dumps(body))//4)}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data);return
            try:
                messages=convert_messages(body)
                tools=[{'type':'function','function':{'name':t['name'],'description':t.get('description',''),'parameters':t.get('input_schema',{})}} for t in body.get('tools',[])]
                result=bridge.completion(messages,tools=tools or None,max_tokens=min(body.get('max_tokens',8192),16000))
                content=[]
                if result.get('content'):content.append({'type':'text','text':result['content']})
                for call in result.get('tool_calls') or []:
                    content.append({'type':'tool_use','id':call['id'],'name':call['function']['name'],'input':json.loads(call['function']['arguments'])})
                usage=result.get('_transport_usage',{})
                response={'id':'msg_'+secrets.token_hex(12),'type':'message','role':'assistant','model':bridge.model,
                    'content':content,'stop_reason':'tool_use' if result.get('tool_calls') else 'end_turn','stop_sequence':None,
                    'usage':{'input_tokens':usage.get('prompt_tokens',0),'output_tokens':usage.get('completion_tokens',0)}}
            except BaseException as exc:
                self.send_response(400);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(json.dumps({'type':'error','error':{'type':'invalid_request_error','message':type(exc).__name__+': '+str(exc)[:200]}}).encode());return
            self.send_response(200)
            self.send_header('Content-Type','text/event-stream' if body.get('stream') else 'application/json')
            self.end_headers()
            if not body.get('stream'):
                self.wfile.write(json.dumps(response).encode());return
            def event(kind,value):
                self.wfile.write(('event: '+kind+'\ndata: '+json.dumps(value)+'\n\n').encode());self.wfile.flush()
            event('message_start',{'type':'message_start','message':dict(response,content=[],stop_reason=None)})
            for index,block in enumerate(content):
                start=dict(block)
                if block['type']=='text':start['text']=''
                else:start['input']={}
                event('content_block_start',{'type':'content_block_start','index':index,'content_block':start})
                delta={'type':'text_delta','text':block['text']} if block['type']=='text' else {'type':'input_json_delta','partial_json':json.dumps(block['input'])}
                event('content_block_delta',{'type':'content_block_delta','index':index,'delta':delta})
                event('content_block_stop',{'type':'content_block_stop','index':index})
            event('message_delta',{'type':'message_delta','delta':{'stop_reason':response['stop_reason'],'stop_sequence':None},'usage':{'output_tokens':response['usage']['output_tokens']}})
            event('message_stop',{'type':'message_stop'})
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}',token
    finally:server.shutdown();server.server_close();thread.join(timeout=2)
