"""Cross frozen public-OOF category alternatives with the retained binary head."""
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.jh_classifier_matrix import metrics,selective
from pipelines.kimjaehak.classifier_models import public_data
from pipelines.kimjaehak.round7_binary_model import reweight

SOURCES={
 'baseline':'jh_r7_hybrid_hybrid1/char_mlp.npz',
 'wordchar':'jh_r7_cv_features1/wordchar_mlp.npz',
 'boundary':'jh_r7_cv_features1/boundary_mlp.npz',
 'wide':'jh_r7_models_alternatives1/char_wide.npz',
 'svm10':'jh_r7_models_alternatives1/char_svm10.npz',
 'blend25':'jh_r7_hybrid_hybrid1/blend_dense0.25.npz',
 'dense':'jh_r7_hybrid_hybrid1/dense_mlp.npz',
 'min1':'jh_r7_regularization_screen1/min1.npz',
 'grams36':'jh_r7_regularization_screen1/grams36.npz'}

def main():
    dest=ROOT/'experiments/.cache/jh_r8_category_cross1';dest.mkdir()
    cache=ROOT/'experiments/.cache';ref=json.loads((cache/'jh_r7_binary_screen1/manifest.json').read_text());binary=cache/'jh_r7_binary_screen1/char_mlp_binary.npz';bp=np.load(binary)['block_probability']
    rows,_,_,_=public_data();assert ref['ids']==[r['id'] for r in rows];gold=np.array([r['labels']['category'] for r in rows]);reports=[]
    for name,source in SOURCES.items():
        path=cache/source;m=json.loads((path.parent/'manifest.json').read_text())
        if any(m[k]!=ref[k] for k in ['ids','folds']):raise ValueError('Reference identity mismatch')
        z=np.load(path);p=reweight(z['probabilities'],z['classes'],bp,1.);pred=z['classes'][p.argmax(1)]
        r={'model':name,'source':source,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,p.max(1))}
        np.savez_compressed(dest/(name+'.npz'),classes=z['classes'],probabilities=p)
        reports.append(r);print(name,r['matched_coverage']['0.5']['category_errors'],r['metrics']['block_f1'],flush=True)
    result={'scope':'posthoc_public_OOF_factorial_screen_not_holdout','api_calls':0,
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'results':reports}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');(dest/'COMPLETE').write_text('Offline factorial screen complete.\n')
if __name__=='__main__':main()
