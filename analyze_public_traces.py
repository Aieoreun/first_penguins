"""Offline scoring and stage attribution of public-only traces; no LLM calls."""
import csv,json,sys,statistics
from collections import Counter
from pathlib import Path
PROJECT=Path('/Users/peace/Downloads/first_penguins-main');sys.path.insert(0,str(PROJECT))
from evaluate_guard_review import score_run,compare_runs
from harness.score_local import as_scoring_input
import scoring

def errors(p,response):
 d=scoring.score_guard(*as_scoring_input(p),response)
 result=[]
 if not d['valid']:return ['invalid_response'],d
 if not d['input_correct']:result.append('decision')
 if d['category_scored'] and not d['category_correct']:result.append('category')
 if d['has_output']:
  if not d['output_correct']:result.append('grounded')
  if not d['safe_correct']:result.append('safe')
  if d['span_f1']<1:result.append('unsupported')
 return result,d

def main(directory):
 directory=Path(directory);public=[json.loads(l) for l in (PROJECT/'public/guard.jsonl').read_text().splitlines() if l.strip()]
 traces={}
 for p in (directory/'traces').glob('*.json'):
  r=json.loads(p.read_text())
  if r['id'] in traces:raise ValueError('duplicate_trace_id_for_item')
  traces[r['id']]=(r,p)
 assert len(traces)==200
 http={r['id']:r for r in json.loads((directory/'http-responses.json').read_text())}
 rows=[];initial_records=[];final_records=[];metrics=Counter();actions=Counter();stages=Counter();stage_errors=Counter();times={};harm=[];fix=[];observations=[];new_error_origins=Counter()
 for p in public:
  trace,path=traces[p['id']];events=trace['events']
  initial_event=next((e for e in events if e['event']=='initial_verdict'),None)
  initial=initial_event['response'] if initial_event else None;final=trace.get('final_response')
  assert final==http[p['id']]['response']
  init_record={'response':initial,'error':None if initial else trace.get('error_type','missing_initial'),'seconds':initial_event['at_seconds'] if initial_event else trace['seconds']}
  final_record=http[p['id']];initial_records.append(init_record);final_records.append(final_record)
  before,bs=errors(p,initial);after,fs=errors(p,final)
  metrics.update(trace['metrics'])
  for e in events:
   if e['event']=='supervisor_action':actions[(e['target'],e['action'],e.get('reason',''))]+=1
   if e['event']=='stage_start':stages[e['stage']]+=1
   if e['event']=='stage_error':stage_errors[(e['stage'],e['error_type'])]+=1
   if e['event']=='stage_end':times.setdefault(e['stage'],[]).append(e['seconds'])
  new=set(after)-set(before);resolved=set(before)-set(after)
  attribution=[]
  for field in sorted(new-{'unsupported'}):
   target='input' if field in ['decision','category'] else 'grounding' if field=='grounded' else 'safety'
   relevant_actions=[e for e in events if e['event']=='supervisor_action' and e.get('target')==target]
   parsed=[e for e in events if e['event']=='stage_result' and e['stage']==('input_review' if target=='input' else 'output_review')]
   rechecks=[e for e in events if e['event']=='recheck_verdict' and e.get('target')==target]
   origin='input_review_applied' if target=='input' and any(e['action']=='apply' for e in relevant_actions) else target+'_recheck_applied' if any(e['action']=='apply' for e in relevant_actions) else 'unresolved_attribution'
   new_error_origins[origin]+=1
   detail={'id':p['id'],'field':field,'gold':p['labels'].get(field),'initial':initial,'final':final,'messages':p['messages'],'candidate_answer':p.get('candidate_answer'),'origin':origin,'review_reports':parsed,'supervisor_actions':relevant_actions,'recheck_verdicts':rechecks,'trace_file':str(path)}
   harm.append(detail);attribution.append({'field':field,'origin':origin})
  for field in sorted(resolved-{'unsupported'}):fix.append({'id':p['id'],'field':field,'initial':initial,'final':final,'trace_file':str(path)})
  row={'id':p['id'],'last_user_message':next((m['content'] for m in reversed(p['messages']) if m['role']=='user'),''),'candidate_answer':p.get('candidate_answer'),'labels':p['labels'],'initial':initial,'final':final,'initial_errors':before,'final_errors':after,'new_errors':sorted(new),'resolved_errors':sorted(resolved),'initial_span_f1':bs['span_f1'],'final_span_f1':fs['span_f1'],'initial_seconds':init_record['seconds'],'http_seconds':final_record['seconds'],'supervisor_actions':[{'target':e['target'],'action':e['action'],'reason':e.get('reason')} for e in events if e['event']=='supervisor_action'],'new_error_attribution':attribution,'trace_file':str(path)}
  rows.append(row)
 summary={'public_items':200,'initial':score_run(public,initial_records),'final':score_run(public,final_records),'changes':compare_runs(public,initial_records,final_records),'metrics':dict(metrics),'stage_calls':dict(stages),'stage_errors':[{'stage':k[0],'type':k[1],'count':v} for k,v in stage_errors.items()],'stage_mean_seconds':{k:round(statistics.mean(v),3) for k,v in times.items()},'actions':[{'target':k[0],'action':k[1],'reason':k[2],'count':v} for k,v in actions.items()],'new_core_error_origins':dict(new_error_origins),'new_core_error_events':len(harm),'new_core_error_items':len({r['id'] for r in harm}),'fixed_core_error_events':len(fix),'fixed_core_error_items':len({r['id'] for r in fix}),'initial_core_error_items':sum(any(e!='unsupported' for e in r['initial_errors']) for r in rows),'final_core_error_items':sum(any(e!='unsupported' for e in r['final_errors']) for r in rows),'unchanged_wrong_core_items':sum(bool((set(r['initial_errors'])&set(r['final_errors']))-{'unsupported'}) for r in rows),'span_improved_items':sum(r['initial_span_f1'] is not None and r['final_span_f1']>r['initial_span_f1'] for r in rows),'span_worsened_items':sum(r['initial_span_f1'] is not None and r['final_span_f1']<r['initial_span_f1'] for r in rows),'trace_files':len(traces),'trace_equals_http':True,'initial_mean_seconds':statistics.mean(r['seconds'] for r in initial_records),'final_http_mean_seconds':statistics.mean(r['seconds'] for r in final_records),'limitations':['single public-set run','review observation semantic correctness is not labeled directly','stage event timings include monitoring overhead; initial timings exclude HTTP','new traces are not a reconstruction of previous HTTP errors']}
 summary['changes']['category']={direction:sum(('category' in r['initial_errors']) and ('category' not in r['final_errors']) if direction=='wrong_to_right' else ('category' not in r['initial_errors']) and ('category' in r['final_errors']) for r in rows) for direction in ['wrong_to_right','right_to_wrong']}
 for name,data in [('stage-summary.json',summary),('new-errors.json',harm),('fixed-errors.json',fix)]:
  (directory/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
 (directory/'items.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 with (directory/'items.csv').open('w',newline='') as f:
  keys=list(rows[0]);w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
 lines=['# 공개 200개 단계별 요약','', '| 문항 | 1차 오류 | 최종 오류 | 새 오류 | 해결 | 감독자 조치 | HTTP초 |','|---|---|---|---|---|---|---:|']
 for r in rows:
  join=lambda xs:', '.join(xs) or '없음'
  act='; '.join(x['target']+':'+x['action']+'('+str(x['reason'])+')' for x in r['supervisor_actions'])
  lines.append(f"| {r['id']} | {join(r['initial_errors'])} | {join(r['final_errors'])} | {join(r['new_errors'])} | {join(r['resolved_errors'])} | {act} | {r['http_seconds']:.2f} |")
 (directory/'items.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main(sys.argv[1])
