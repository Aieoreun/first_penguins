"""Opt-in PUBLIC-only diagnostic app. Never imported by server:app."""
import asyncio,contextvars,copy,functools,json,os,secrets,sys,time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

PROJECT=Path(os.environ.get('GUARD_PROJECT_ROOT',Path(__file__).resolve().parent)).resolve()
sys.path.insert(0,str(PROJECT))
if os.environ.get('GUARD_PUBLIC_DIAGNOSTICS')!='1':
 raise RuntimeError('public_diagnostics_not_enabled')
TRACE_DIR=Path(os.environ['GUARD_PUBLIC_TRACE_DIR']).resolve()
if not TRACE_DIR.is_relative_to(PROJECT/'work/evaluations'):
 raise RuntimeError('trace_directory_must_be_public_evaluation_directory')
TRACE_DIR.mkdir(parents=True,exist_ok=True)
PUBLIC={}
def request_key(messages,candidate):
 return json.dumps([messages,candidate],ensure_ascii=False,sort_keys=True,separators=(',',':'))
for line in (PROJECT/'public/guard.jsonl').read_text().splitlines():
 if not line.strip():continue
 row=json.loads(line)
 if row.get('endpoint')!='guard':raise RuntimeError('non_guard_public_record')
 key=request_key(row['messages'],row.get('candidate_answer'))
 if key in PUBLIC:raise RuntimeError('ambiguous_public_request')
 PUBLIC[key]={'id':row['id'],'messages':row['messages'],'candidate_answer':row.get('candidate_answer')}
if len(PUBLIC)!=200:raise RuntimeError('unexpected_public_set')

import guard_supervisor as gs
import server
ACTIVE=contextvars.ContextVar('public_trace')
STAGE=contextvars.ContextVar('public_stage',default='supervisor')

def emit(event,**data):
 row=ACTIVE.get()
 item={'seq':len(row['events']),'at_seconds':time.monotonic()-row['_start'],'stage':STAGE.get(),'event':event,**copy.deepcopy(data)}
 row['events'].append(item)

def stage_wrapper(name,original):
 @functools.wraps(original)
 async def wrapped(*args,**kwargs):
  stage=name
  if name in {'base_input','base_grounding','base_safety'} and kwargs.get('review_feedback') is not None:
   stage=name.replace('base_','recheck_')
  token=STAGE.set(stage);start=time.monotonic();emit('stage_start')
  try:
   value=await original(*args,**kwargs)
   emit('stage_result',response=value.model_dump())
   return value
  except BaseException as error:
   emit('stage_error',error_type=type(error).__name__)
   raise
  finally:
   emit('stage_end',seconds=time.monotonic()-start);STAGE.reset(token)
 return wrapped

for function,stage in [('classify_input','base_input'),('check_grounding','base_grounding'),('check_safety','base_safety'),('review_input','input_review'),('review_verdict','output_review'),('review_input_status','input_review'),('review_grounding','grounding_review'),('review_safety','safety_review')]:
 setattr(gs,function,stage_wrapper(stage,getattr(gs,function)))

class DiagnosticSearch:
 def __init__(self,real):self.real=real
 async def search(self,query,top_k=5):
  start=time.monotonic();emit('search_request',kind='initial' if top_k==5 else 'research',query=query,top_k=top_k)
  try:
   evidence=await self.real.search(query,top_k=top_k)
   emit('search_result',kind='initial' if top_k==5 else 'research',evidence=evidence,seconds=time.monotonic()-start)
   return evidence
  except BaseException as error:
   emit('search_error',error_type=type(error).__name__,seconds=time.monotonic()-start);raise

class PublicOnlyGuard:
 def __init__(self,original):self.original=original
 async def evaluate(self,messages,candidate_answer=None):
  public=PUBLIC.get(request_key(messages,candidate_answer))
  if public is None:raise ValueError('request_not_in_public_set')  # no recording or API call
  row={**copy.deepcopy(public),'trace_id':secrets.token_hex(8),'events':[],'_start':time.monotonic()}
  active=ACTIVE.set(row)
  async def create(**kwargs):
   # Explicit allowlist: no headers, credentials, SDK objects or exception bodies.
   allowed=('model','messages','response_format','max_completion_tokens','extra_body')
   emit('llm_request',request={k:copy.deepcopy(kwargs[k]) for k in allowed if k in kwargs})
   start=time.monotonic()
   try:
    response=await self.original.client.chat.completions.create(**kwargs)
    choice=response.choices[0]
    emit('llm_response',content=choice.message.content,finish_reason=choice.finish_reason,
         usage=response.usage.model_dump() if response.usage else {},seconds=time.monotonic()-start)
    return response
   except BaseException as error:
    emit('llm_error',error_type=type(error).__name__,seconds=time.monotonic()-start);raise
  proxy=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  supervisor=gs.GuardSupervisor(proxy,self.original.model,DiagnosticSearch(self.original.searcher),review_mode='all',
     total_timeout=self.original.total_timeout,base_timeout=self.original.base_timeout,review_timeout=self.original.review_timeout,observer=emit)
  try:
   result=await supervisor.evaluate(messages,candidate_answer)
   row['final_response']=copy.deepcopy(result);emit('final_response',response=result)
   return result
  except BaseException as error:
   row['error_type']=type(error).__name__;emit('request_error',error_type=type(error).__name__);raise
  finally:
   row['seconds']=time.monotonic()-row.pop('_start');row['metrics']=dict(supervisor.metrics)
   ACTIVE.reset(active)
   # Only known-public requests reach this path. Every invocation gets its own file.
   target=TRACE_DIR/(row['id']+'--'+row['trace_id']+'.json')
   target.write_text(json.dumps(row,ensure_ascii=False,indent=2))

app=server.app
_original_lifespan=app.router.lifespan_context
@asynccontextmanager
async def diagnostic_lifespan(application):
 async with _original_lifespan(application):
  application.state.guard=PublicOnlyGuard(application.state.guard)
  yield
app.router.lifespan_context=diagnostic_lifespan
