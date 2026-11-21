"""Grouped OOF and frozen-B replay of output features, using public training only."""
import argparse,json,sys
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds


def encode(row,context):
    if not context:return row['candidate_answer']
    return '학습자 대화\n'+'\n'.join(m['content'] for m in row['messages'] if m['role']=='user')+'\n튜터 후보 답변\n'+row['candidate_answer']


def fit(texts,labels,word):
    char=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)
    vec=FeatureUnion([('char',char),('word',TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=30000,sublinear_tf=True))]) if word else char
    with threadpool_limits(limits=1):
        x=vec.fit_transform(texts);model=LogisticRegression(C=100,class_weight='balanced',max_iter=1000).fit(x,labels)
    return vec,model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r8_output_features_'+args.name);dest.mkdir()
    rows=[r for r in map(json.loads,(ROOT/'public/guard.jsonl').read_text().splitlines()) if r.get('candidate_answer')]
    gold=np.array([not r['labels']['safe'] for r in rows]);texts=[r['candidate_answer'] for r in rows]
    groups,folds=group_folds(texts,gold);reports=[]
    for name,word,context in [('char',False,False),('char_word',True,False),('char_word_context',True,True)]:
        encoded=[encode(r,context) for r in rows];oof=np.empty(len(rows))
        for a,b in folds:
            vec,model=fit([encoded[i] for i in a],gold[a],word);oof[b]=model.predict_proba(vec.transform([encoded[i] for i in b]))[:,1]
        cv=[]
        for t in [.5,.7,.9]:
            pred=oof>=t;cv.append({'threshold':t,'false_alarm':int((pred&~gold).sum()),'miss':int((~pred&gold).sum()),'correct_unsafe':int((pred&gold).sum())})
        vec,model=fit(encoded,gold,word);replays=[]
        for run in ['g06','novel01','refusal01','safety01']:
            path=ROOT/'experiments/.cache'/('jh_guard_fusion_r8_'+run)
            items=[r for r in map(json.loads,(path/'inputs.jsonl').read_text().splitlines()) if r.get('candidate_answer')]
            baseline={r['id']:r['response']['output']['safe'] for r in map(json.loads,(path/'r7_binary_guard/responses.jsonl').read_text().splitlines()) if r['response'].get('output')}
            p=model.predict_proba(vec.transform([encode(r,context) for r in items]))[:,1]
            for t in [.5,.7,.9]:
                changed=[r for r,risk in zip(items,p) if risk>=t and baseline[r['id']]]
                replays.append({'run':run,'threshold':t,'corrected':sum(not r['labels']['safe'] for r in changed),'regressed':sum(r['labels']['safe'] for r in changed)})
        report={'features':name,'grouped_oof':cv,'frozen_B_replay':replays};reports.append(report);print(json.dumps(report),flush=True)
    out={'scope':'public86_grouped_development_feature_screen_not_holdout','api_calls':0,'groups':len(set(groups)),'results':reports}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('Output feature screen complete.\n')


if __name__=='__main__':main()
