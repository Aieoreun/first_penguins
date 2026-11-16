"""Fresh public-only diagnostic HTTP server; complete per-item stage traces."""
import json,os,secrets,shutil,socket,subprocess,sys,time,urllib.error,urllib.request
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
PROJECT=Path('/Users/peace/Downloads/first_penguins-main')
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT))
from guard_schemas import validate_guard_response
from evaluate_guard_review import fingerprint,score_run

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**kw):return None

def http(url,payload,token=None,timeout=30):
 headers={'Content-Type':'application/json'}
 if token is not None:headers['Authorization']='Bearer '+token
 request=urllib.request.Request(url,data=json.dumps(payload,ensure_ascii=False).encode(),headers=headers,method='POST')
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
 with opener.open(request,timeout=timeout) as response:
  return response.status,response.read(256*1024+1)

def one(item,url,token):
 payload={'spec_version':'1.0','session_id':'public-http-'+secrets.token_hex(8),'messages':item['messages']}
 candidate=item.get('candidate_answer')
 if candidate is not None:payload['candidate_answer']=candidate
 row={'id':item['id'],'response':None,'error':None,'status':None};start=time.monotonic()
 try:
  status,body=http(url,payload,token);row['status']=status
  if status!=200 or len(body)>256*1024:raise ValueError('invalid_http_response')
  data=json.loads(body);result=validate_guard_response(data,candidate)
  if candidate is None and result.output is not None:raise ValueError('unexpected_output')
  if result.output is not None and result.output.grounded!=(not result.output.unsupported):raise ValueError('inconsistent_grounding')
  row['response']=data
 except urllib.error.HTTPError as error:
  row.update(status=error.code,error=f'HTTPError:{error.code}');error.close()
 except Exception as error:row['error']=type(error).__name__
 row['seconds']=time.monotonic()-start
 return row

def main():
 dest=PROJECT/'work/evaluations'/('public-monitor-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'));dest.mkdir();(dest/'traces').mkdir();(dest/'source').mkdir()
 before=fingerprint()
 for p in PROJECT.glob('*.py'):shutil.copy2(p,dest/'source'/p.name)
 shutil.copy2(__file__,dest/'source/run_public_monitor.py')
 public=[json.loads(l) for l in (PROJECT/'public/guard.jsonl').read_text().splitlines() if l.strip()];assert len(public)==200
 token=os.environ['TEAM_TOKEN'];proc=None;log=(dest/'server.log').open('w');rows={}
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 url=f'http://127.0.0.1:{port}/guard'
 manifest={'model':os.environ['OPENAI_MODEL'],'source_hashes':before,'port':port,'workers':8,'scope':'fresh diagnostic HTTP /guard; exact public allowlist; one base pass per item with linked traces; observer disabled in ordinary server:app'}
 try:
  env=dict(os.environ,GUARD_PUBLIC_DIAGNOSTICS='1',GUARD_PUBLIC_TRACE_DIR=str(dest/'traces'),GUARD_PROJECT_ROOT=str(PROJECT),GUARD_REVIEW_MODE='all',PYTHONDONTWRITEBYTECODE='1')
  proc=subprocess.Popen([sys.executable,'-m','uvicorn','public_guard_monitor:app','--host','127.0.0.1','--port',str(port)],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT);manifest['pid']=proc.pid
  end=time.monotonic()+25
  while True:
   if proc.poll() is not None:raise RuntimeError('server_start_failed')
   try:http(url,{'messages':[]},timeout=1)
   except urllib.error.HTTPError as e:
    code=e.code;e.close()
    if code!=401:raise RuntimeError('auth_failed')
    break
   except (urllib.error.URLError,TimeoutError):
    if time.monotonic()>end:raise TimeoutError('startup')
    time.sleep(.2)
  # Rejected before recording and model invocation; no arbitrary traffic can become a trace.
  try:http(url,{'messages':[{'role':'user','content':'NON_PUBLIC_DIAGNOSTIC_PROBE_47ec'}]},token,timeout=2)
  except urllib.error.HTTPError as e:
   code=e.code;e.close()
   if code!=503:raise RuntimeError('public_allowlist_failed')
  else:raise RuntimeError('non_public_request_accepted')
  assert not list((dest/'traces').glob('*.json'))
  manifest['non_public_request_rejected_without_trace']=True
  (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));print('SERVER_READY',port,flush=True);print('DESTINATION',dest,flush=True)
  with ThreadPoolExecutor(max_workers=8) as pool:
   futures={pool.submit(one,p,url,token):p for p in public}
   for future in as_completed(futures):
    p=futures[future];r=future.result();rows[p['id']]=r
    if len(rows)%20==0:print(f'completed {len(rows)}/200',flush=True)
  records=[rows[p['id']] for p in public]
  (dest/'http-responses.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
  summary=score_run(public,records);summary['mean_seconds']=sum(r['seconds'] for r in records)/len(records);summary['http_statuses']=dict(Counter(str(r['status']) for r in records))
  after=fingerprint();summary['source_changed']=[k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)]
  summary['traces']=len(list((dest/'traces').glob('*.json')))
  (dest/'http-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
 finally:
  if proc and proc.poll() is None:
   proc.terminate()
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
  log.close();(dest/'cleanup.json').write_text(json.dumps({'pid':proc.pid if proc else None,'stopped':proc.poll() is not None if proc else True}))
 print('SAVED',dest,flush=True)
if __name__=='__main__':main()
