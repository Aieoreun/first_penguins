"""Replay recorded public judges consistently; no generation, API or new labels.

Both first-observed and last-observed judge results are reported, never selected
for a favorable candidate. This is a variance diagnostic, not a new live score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.jh_compare import evaluator_hashes, request_key, score_local, load_kb_dir


def key_for(kind,payload):
    return hashlib.sha256(json.dumps([kind,payload],ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def replay(root):
    if not (root/'COMPLETE').exists():raise ValueError('Only completed experiments can be replayed')
    plan=json.loads((root/'plan.json').read_text())
    raw=(root/'inputs.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=plan['dataset_sha256']:raise ValueError('Dataset changed')
    items=[json.loads(s) for s in raw.splitlines()]
    first,last,observed={},{},{}
    for candidate in plan['order']:
        path=root/candidate
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest['evaluator_hashes']!=evaluator_hashes():raise ValueError('Evaluator changed')
        for line in (path/'judge.jsonl').read_text().splitlines():
            row=json.loads(line);key=row['key'];value=row['result']
            first.setdefault(key,value);last[key]=value
            observed.setdefault(key,[]).append(value)
    results=[]
    kb=load_kb_dir(ROOT/'kb')
    for candidate in plan['order']:
        path=root/candidate
        rows=[json.loads(s) for s in (path/'responses.jsonl').read_text().splitlines()]
        by_id={r['id']:r for r in rows}
        if len(by_id)!=len(rows) or set(by_id)!={r['id'] for r in items}:
            raise ValueError('Missing or duplicate responses')
        responses={request_key(p['endpoint'],p):by_id[p['id']]['response'] for p in items}
        def call(endpoint,payload):return responses[request_key(endpoint,payload)]
        original=json.loads((path/'result.json').read_text())
        entry={'candidate':candidate,'live_tutor':original['tutor'],'replayed':{}}
        for name,cache in (('first_observed',first),('last_observed',last)):
            missing=[]
            def judge(kind,payload):
                key=key_for(kind,payload)
                if key not in cache:missing.append(key);raise KeyError('Unrecorded judge call')
                return cache[key]
            result=score_local(items,call,judge,kb,workers=1)
            if missing:raise ValueError('Missing recorded judgment; refusing partial replay')
            entry['replayed'][name]={k:result[k] for k in ('tutor','guard','total','invalid','judge_errors')}
        results.append(entry)
    repeated=[values for values in observed.values() if len(values)>1]
    return {'run':root.name,'dataset_sha256':plan['dataset_sha256'],'order':plan['order'],
            'identical_judge_payloads_repeated':len(repeated),
            'repeated_payloads_with_differing_judgments':sum(any(v!=values[0] for v in values[1:]) for values in repeated),
            'candidates':results}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    roots=[]
    for name in args.runs:
        path=(ROOT/'experiments/.cache'/name).resolve()
        if path.parent!=(ROOT/'experiments/.cache').resolve():raise ValueError('Invalid local run')
        roots.append(path)
    report={'scope':'fixed_public_judgment_counterfactual_not_live_not_private',
            'api_calls':0,'evaluation_code_unchanged':True,'runs':[replay(p) for p in roots]}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for r in report['runs']:
        print(r['run'],'disagreements',r['repeated_payloads_with_differing_judgments'])
        for c in r['candidates']:
            print(c['candidate'],c['live_tutor'],c['replayed']['first_observed']['tutor'],c['replayed']['last_observed']['tutor'])


if __name__=='__main__':main()
