"""Replay frozen public-trained normal probabilities; no fit or API calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def screen(source):
    source = source.resolve()
    if not source.is_relative_to(ROOT/'experiments/.cache'):
        raise ValueError('Expected existing local CPU screen')
    metadata = json.loads((source/'manifest.json').read_text())
    if metadata['scope'] != 'offline_tutor_gate_cpu_development_not_holdout':
        raise ValueError('Unexpected source scope')
    for spec in metadata['sources'].values():
        if hashlib.sha256((ROOT/spec['path']).read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Source dataset changed')
    if metadata['classifier']['C'] != 1000 or metadata['classifier']['threshold'] != .95:
        raise ValueError('Expected original C1000 threshold .95 model')
    records = [r for r in map(json.loads,(source/'predictions.jsonl').read_text().splitlines())
               if r['mode']=='raw' and r['original_endpoint']=='tutor']
    result = {}
    for scope in ('public_oof','primary388','alternative365'):
        rows = [r for r in records if r['scope']==scope]
        result[scope] = {'count':len(rows),'thresholds':{}}
        for cutoff in (.94,.92,.90):
            extra = [r for r in rows if r['prediction']=='normal' and cutoff<=r['confidence']<.95]
            result[scope]['thresholds'][str(cutoff)] = {
                'newly_routed':len(extra),'gold_categories':dict(Counter(r['gold'] for r in extra)),
                'items':[{k:r[k] for k in ('id','gold','confidence')} for r in extra]}
    return {'scope':'development_normal_gate_probability_replay_not_http',
            'source':str(source.relative_to(ROOT)),
            'prediction_sha256':hashlib.sha256((source/'predictions.jsonl').read_bytes()).hexdigest(),
            'no_fit_or_api':True,'runtime_uses_item_ids_or_gold':False,
            'limitations':['Cutoff selected on development evidence, not independent validation.',
                           'PII category differs from normal although tutor action is answer.',
                           'Only newly routed items are counted, not all gate errors.'],
            'datasets':result}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=screen(args.source)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for scope,rows in report['datasets'].items():
        print(scope,rows['count'],{k:v['gold_categories'] for k,v in rows['thresholds'].items()})
