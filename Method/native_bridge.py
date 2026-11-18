"""Selected-model transport for native controllers; no replacement agent loop."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import threading
import time
import urllib.request
import urllib.error

class NativeBudgetExceeded(BaseException):
    pass


class Bridge:
    def __init__(self,backend,key,output,max_calls=64):
        self.backend,self.key,self.output,self.max_calls=backend,key,Path(output),max_calls
        self.model='MiniMax-M3' if backend=='minimax' else 'gpt-5.6-sol'
        self.lock=threading.Lock()
        self.count=0
        self.output.mkdir(parents=True,exist_ok=True)

    def completion(self,messages,tools=None,max_tokens=8192,tool_choice=None,_schema_retry=0):
        with self.lock:
            if self.count>=self.max_calls: raise NativeBudgetExceeded('native model-call budget exhausted')
            index=self.count
            self.count+=1
        record={'call':index,'model':self.model,'backend':self.backend,'messages':messages,'tools':tools,
                'thinking':'disabled' if self.backend=='minimax' else 'low','tool_choice':tool_choice}
        started=time.monotonic()
        try:
            if self.backend=='minimax':
                if not self.key: raise RuntimeError('MiniMax key not loaded')
                payload={'model':self.model,'messages':messages,'max_tokens':max_tokens,'temperature':1.0,
                         'thinking':{'type':'disabled'},'stream':False}
                if not any(m.get('role')=='user' for m in messages):
                    payload['messages']=[*messages,{'role':'user','content':'Follow the supplied instructions and return the requested response.'}]
                    record['system_only_message_adaptation']=True
                if tools:payload['tools']=tools
                if tool_choice:payload['tool_choice']=tool_choice
                request=urllib.request.Request('https://api.minimaxi.com/v1/chat/completions',
                    data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
                record['http_attempts']=[]
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(request,timeout=180) as reply:body=json.load(reply)
                        record['http_attempts'].append({'attempt':attempt+1,'status':200})
                        break
                    except urllib.error.HTTPError as exc:
                        try:
                            error=json.loads(exc.read())
                            detail=error.get('error',{}).get('message') or error.get('base_resp',{}).get('status_msg') or 'request rejected'
                        except Exception:detail='request rejected'
                        detail=str(detail).replace(self.key,'[REDACTED]')[:300]
                        record['http_attempts'].append({'attempt':attempt+1,'status':exc.code,'error':detail})
                        if exc.code in {429,500,502,503,504,529} and attempt<2:
                            time.sleep(2**attempt)
                            continue
                        raise RuntimeError(f'MiniMax HTTP {exc.code}: {detail}') from None
                if body.get('model')!=self.model:raise RuntimeError('unexpected model response')
                result=body['choices'][0]['message']
                record['usage']=body.get('usage',{})
                record['finish_reason']=body['choices'][0].get('finish_reason')
            else:
                from scictrl.api_backends import chat,json_answer
                if tools:
                    schema={'type':'object','properties':{'content':{'type':['string','null']},
                        'tool_calls':{'type':'array','items':{'type':'object','properties':{'id':{'type':'string'},
                            'type':{'type':'string','enum':['function']},'function':{'type':'object','properties':{
                                'name':{'type':'string'},'arguments':{'type':'string'}},'required':['name','arguments'],'additionalProperties':False}},
                            'required':['id','type','function'],'additionalProperties':False}}},
                        'required':['content','tool_calls'],'additionalProperties':False}
                    contract={'role':'system','content':
                        'You are the model backend of an external agent controller. Do not use Codex tools. '
                        'Choose from the supplied function definitions by returning ONLY an OpenAI assistant-message JSON object '
                        'with content (string or null), tool_calls (list; empty when answering). Each tool call has '
                        'id, type="function", function={name,arguments:JSON-encoded string}. '
                        'The external controller will execute these calls and supply tool results. '
                        'Do not claim a tool has run before its result is supplied. Available tools: '+json.dumps(tools)+
                        '\nRequired tool choice, when non-null: '+json.dumps(tool_choice)}
                    reply=chat('codex',[contract,*messages],timeout=180,max_tokens=max_tokens,reasoning_effort='low',output_schema=schema)
                    record['raw_response_text']=reply['content']
                    result=json_answer(reply['content'])
                    if not isinstance(result.get('tool_calls',[]),list):raise ValueError('invalid tool calls')
                    for call in result.get('tool_calls',[]):
                        arguments=call['function']['arguments']
                        if isinstance(arguments,dict):call['function']['arguments']=json.dumps(arguments)
                else:
                    reply=chat('codex',messages,timeout=180,max_tokens=max_tokens,reasoning_effort='low')
                    result={'role':'assistant','content':reply['content']}
                record['usage']=reply.get('usage',{})
                record['finish_reason']='tool_calls' if result.get('tool_calls') else 'stop'
            if tools and result.get('tool_calls'):
                import jsonschema
                definitions={t['function']['name']:t['function'].get('parameters',{}) for t in tools if t.get('type')=='function'}
                try:
                    for call in result['tool_calls']:
                        name=call['function']['name']
                        arguments=call['function']['arguments']
                        arguments=json.loads(arguments) if isinstance(arguments,str) else arguments
                        if name in definitions:jsonschema.validate(arguments,definitions[name])
                except (json.JSONDecodeError,jsonschema.ValidationError) as exc:
                    detail=str(exc)[:300]
                    record.update(status='invalid_tool_schema',response=result,error=detail)
                    if _schema_retry>=1:raise ValueError('tool response schema invalid after one serialization retry: '+detail)
                    record['serialization_retry']=True
                    repair={'role':'user','content':'Transport-format validation failed: '+detail+
                        '\nReturn the same intended response with all required tool fields correctly serialized. '
                        'Do not change the scientific decision or claim to execute a tool. Previous malformed response: '+json.dumps(result)}
                    return self.completion([*messages,repair],tools=tools,max_tokens=max_tokens,tool_choice=tool_choice,_schema_retry=1)
            record.update(status='completed',response=result)
            result['_transport_usage']=record.get('usage',{})
            result['_transport_finish_reason']=record.get('finish_reason','stop')
            return result
        except Exception as exc:
            record.update(status='failed',error=type(exc).__name__+': '+str(exc)[:250])
            raise
        finally:
            record['elapsed_s']=round(time.monotonic()-started,3)
            text=json.dumps(record,indent=2,ensure_ascii=False)
            if self.key:text=text.replace(self.key,'[REDACTED]')
            (self.output/f'{index:04d}.json').write_text(text)


def langchain_model(bridge):
    from typing import Any
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration,ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai.chat_models.base import _convert_message_to_dict
    class NativeModel(BaseChatModel):
        transport: Any
        model_name: str
        @property
        def _llm_type(self):return 'scictrl-native-transport'
        def bind_tools(self,tools,**kwargs):
            return self.bind(tools=[convert_to_openai_tool(t) for t in tools])
        def _generate(self,messages,stop=None,run_manager=None,**kwargs):
            wire=[_convert_message_to_dict(m) for m in messages]
            result=self.transport.completion(wire,tools=kwargs.get('tools'))
            calls=[]
            for item in result.get('tool_calls') or []:
                args=item['function']['arguments']
                calls.append({'name':item['function']['name'],'args':json.loads(args) if isinstance(args,str) else args,
                              'id':item['id'],'type':'tool_call'})
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result.get('content') or '',tool_calls=calls))])
        async def _agenerate(self,messages,stop=None,run_manager=None,**kwargs):
            return await asyncio.to_thread(self._generate,messages,stop,run_manager,**kwargs)
    return NativeModel(transport=bridge,model_name=bridge.model)
