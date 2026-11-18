"""Loopback OpenAI protocol for native multiprocessing controllers."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import secrets
import threading
import time

@contextmanager
def serve(bridge):
    token=secrets.token_hex(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.headers.get('Authorization')!='Bearer '+token:
                self.send_error(401)
                return
            try:
                request=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
                if request.get('stream'):raise ValueError('streaming transport unsupported')
                if bridge.backend=='codex' and any(isinstance(m.get('content'),list) for m in request['messages']):
                    raise ValueError('multimodal native stage unsupported by this text transport')
                result=bridge.completion(request['messages'],tools=request.get('tools'),
                    max_tokens=min(request.get('max_tokens') or 8192,16000),tool_choice=request.get('tool_choice'))
                usage=result.pop('_transport_usage',{})
                body={'id':'native-bridge-'+secrets.token_hex(8),'object':'chat.completion','created':int(time.time()),
                      'model':bridge.model,'choices':[{'index':0,'message':result,'finish_reason':'tool_calls' if result.get('tool_calls') else 'stop'}],
                      'usage':{'prompt_tokens':usage.get('prompt_tokens',0),'completion_tokens':usage.get('completion_tokens',0),'total_tokens':usage.get('total_tokens',0)}}
                status=200
            except BaseException as exc:
                # 400 avoids provider retry storms at exhausted local budgets.
                status=400
                body={'error':{'message':type(exc).__name__+': '+str(exc)[:150],'type':'native_transport_error'}}
            data=json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/v1',token
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
