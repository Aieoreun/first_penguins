"""CPU-only screen of last-question weighting and teammate content bigrams."""
import argparse,ast,hashlib,json,re,statistics,subprocess,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'core/schema.py').exists())
sys.path.insert(0,str(ROOT))
from experiments.jh_kiwi_normalization_screen import FixedBM25,SOURCES,EXPECTED,OLD_CODE,normalize_view
from core.kb import load_chunks
from core.search import tokens
from kiwipiepy import Kiwi
REF='de664f2'

def main(output, gated=False):
 output=output.resolve()
 if not output.is_relative_to(ROOT/'experiments/.cache') or output.exists():raise ValueError('Use new local cache output')
 source=subprocess.check_output(['git','show',REF+':app/kb_index.py'],cwd=ROOT,text=True)
 tree=ast.parse(source)
 terms=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='_COURSE_TERMS')
 kiwi=Kiwi(num_workers=1)
 for term in terms:kiwi.add_user_word(term.replace(' ',''),'NNP',0)
 def kt(text):
  text=normalize_view(text)
  out=[t.form.strip().lower() for t in kiwi.tokenize(text) if not t.tag.startswith('S') and t.form.strip()]
  for match in OLD_CODE.findall(text):
   if match.lower() not in out:out.append(match.lower())
  content=[t for t in out if len(t)>=2 or t.isdigit()]
  return out+[a+'_'+b for a,b in zip(content,content[1:])]
 chunks=load_chunks(ROOT/'kb');text=[c.text for c in chunks]
 from pipelines.kimjaehak.learned_sections import train_section_classifier
 vectorizer,model=train_section_classifier(1000)
 indices={'char':FixedBM25(text,tokens),'kiwi_bigram':FixedBM25(text,kt)}
 result={'scope':'public_development_raw_lexical_retrieval_not_http_score','team_ref':subprocess.check_output(['git','rev-parse',REF],cwd=ROOT,text=True).strip(),'team_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'gated':gated,'modes':{}}
 for name,path in SOURCES.items():
  p=ROOT/path;raw=p.read_bytes();rows=[r for r in map(json.loads,raw.decode().splitlines()) if len(r['labels'].get('evidence') or [])==1]
  assert len(rows)==EXPECTED[name]
  meta=p.parent/'derivation.json'
  if meta.exists():
   provenance=json.loads(meta.read_text());assert provenance['dataset_sha256']==hashlib.sha256(raw).hexdigest()
  records=[]
  for row in rows:
   full='\n'.join(m['content'] for m in row['messages']);last=next(m['content'] for m in reversed(row['messages']) if m['role']=='user')
   confidence=float(model.predict_proba(vectorizer.transform([last]))[0].max())
   anaphora=bool(re.search(r'앞서|방금|이어서|계속|(?:그|이|저)\s*(?:거|것|내용|예제|코드|규칙|함수|상품|질문)',last))
   for token_mode,index in indices.items():
    fs=index.scores(full);ls=index.scores(last)
    for weighting in (('full','confidence25','confidence50','no_anaphora') if gated else ('full','raw75','normalized75_gate12')):
     scores=fs
     if weighting=='raw75':scores=[.75*a+.25*b for a,b in zip(ls,fs)]
     elif (weighting=='normalized75_gate12' or (weighting=='confidence25' and confidence>=.25) or (weighting=='confidence50' and confidence>=.50) or (weighting=='no_anaphora' and not anaphora)) and max(ls,default=0)>=12:
      lm=max(ls) or 1;fm=max(fs) or 1;scores=[.75*a/lm+.25*b/fm for a,b in zip(ls,fs)]
     ranked=[i for i in sorted(range(len(scores)),key=lambda i:-scores[i]) if scores[i]>0][:10]
     sections=list(dict.fromkeys(chunks[i].section_id for i in ranked));gold=row['labels']['evidence'][0]
     records.append({'id':row['id'],'mode':token_mode+'_'+weighting,'hits':[gold in sections[:k] for k in (1,3,6)]})
  counts={}
  for mode in sorted({r['mode'] for r in records}):
   rs=[r for r in records if r['mode']==mode];counts[mode]=[sum(r['hits'][k] for r in rs) for k in range(3)]
  result['modes'][name]={'count':len(rows),'dataset_sha256':hashlib.sha256(raw).hexdigest(),'top1_top3_top6':counts}
  print(name,len(rows),counts,flush=True)
 output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--gated',action='store_true');args=ap.parse_args();main(args.output,args.gated)
