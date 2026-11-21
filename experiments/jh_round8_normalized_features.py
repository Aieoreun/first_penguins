"""Public grouped OOF NFKC/whitespace normalization of category/binary heads."""
import json,hashlib,sys,re,unicodedata
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pipelines.kimjaehak.classifier_models import CategoryModel,public_data
from pipelines.kimjaehak.round7_binary_model import BLOCKED,reweight
from experiments.jh_classifier_matrix import group_folds,metrics,selective


def vectorizer(**kwargs):
    return TfidfVectorizer(**kwargs,preprocessor=lambda text:re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text).replace('\u200b','').replace('\ufeff','')).strip().lower())


def main():
    cache=ROOT/'experiments/.cache';dest=cache/'jh_r8_normalized_features1';dest.mkdir()
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows]);binary=np.array(['block' if c in BLOCKED else 'allow' for c in gold]);groups,folds=group_folds(texts,gold)
    paths=[cache/'jh_r7_hybrid_hybrid1',cache/'jh_r7_binary_screen1'];fs=[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]
    for path in paths:
        manifest=json.loads((path/'manifest.json').read_text());assert manifest['ids']==[r['id'] for r in rows] and manifest['folds']==fs
    old=np.load(paths[0]/'char_mlp.npz');classes=old['classes'];cp={'char':old['probabilities']};bp={'char':np.load(paths[1]/'char_mlp_binary.npz')['block_probability']};fits=[]
    newcp=np.empty_like(cp['char']);newbp=np.empty(len(rows))
    for f,(a,b) in enumerate(folds):
        with patch('pipelines.kimjaehak.classifier_models.TfidfVectorizer',side_effect=vectorizer):
            cat=CategoryModel('char_mlp').fit([texts[i] for i in a],gold[a]);decision=CategoryModel('char_mlp').fit([texts[i] for i in a],binary[a])
        assert np.array_equal(cat.classes,classes)
        newcp[b]=cat.probabilities([texts[i] for i in b]);newbp[b]=decision.probabilities([texts[i] for i in b])[:,int(np.flatnonzero(decision.classes=='block')[0])]
        fits.append({'fold':f,'category':cat.diagnostics,'binary':decision.diagnostics});print('fold',f,flush=True)
    cp['normalized']=newcp;bp['normalized']=newbp;results=[]
    for c in ['char','normalized']:
        for b in ['char','normalized']:
            p=reweight(cp[c],classes,bp[b],1.);pred=classes[p.argmax(1)];name=c+'_'+b
            report={'candidate':name,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,p.max(1))};results.append(report);print(json.dumps(report),flush=True)
            np.savez_compressed(dest/(name+'.npz'),probabilities=p,classes=classes)
    out={'scope':'public400_grouped_OOF_normalized_feature_factorial_not_independent_holdout','api_calls':0,'ids':[r['id'] for r in rows],'folds':fs,'fits':fits,'results':results}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('Normalization feature screen complete.\n')


if __name__=='__main__':main()
