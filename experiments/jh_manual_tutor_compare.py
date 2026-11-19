"""Compare frozen, manually labeled public-KB tutor development questions.

This runner deliberately does not claim an independent review or holdout. The
proposal stays in the ignored development cache and is never loaded by a server.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from jh_compare import ROOT, score_rows
from experiments.jh_environment import load_environment
from experiments import jh_guard_fusion_compare as runner

SCOPE = 'manual_public_kb_tutor_development_not_private_not_holdout'


def load_proposal(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT/'experiments/.cache').resolve()):
        raise ValueError('Development proposal must be in the ignored cache')
    proposal = json.loads(path.read_text())
    if (proposal.get('scope') != SCOPE or proposal.get('independent_review') is not False
            or proposal.get('model_predictions_inspected_during_design') is not False):
        raise ValueError('Missing honest design provenance')
    actual = {str(p.relative_to(ROOT)):runner.digest(p.read_bytes())
              for p in sorted((ROOT/'kb').rglob('*')) if p.is_file()}
    if actual != proposal.get('kb_hashes'):
        raise ValueError('KB changed since question design')
    items = proposal['items']
    ids = [r['id'] for r in items]
    if not items or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate questions')
    for row in items:
        if row['endpoint'] != 'tutor' or not row['messages']:
            raise ValueError('Only ordinary tutor messages are supported')
        if any(m['role'] not in {'user','assistant'} or not isinstance(m['content'],str)
               for m in row['messages']):
            raise ValueError('Invalid conversation')
        labels = row['labels']
        if not isinstance(labels.get('gold_facts'),list) or not isinstance(labels.get('evidence'),list):
            raise ValueError('Invalid labels')
        if (labels['category'],labels['action']) not in {('normal','answer'),('out_of_material','abstain')}:
            raise ValueError('Unexpected manual category')
        if labels['category']=='normal' and (not labels['gold_facts'] or not labels['evidence']):
            raise ValueError('Supported questions need facts and sources')
        if labels['category']=='out_of_material' and (labels['gold_facts'] or labels['evidence']):
            raise ValueError('Unsupported questions cannot carry gold facts')
        for support in row['support']:
            source = (ROOT/support['path']).resolve()
            if not source.is_relative_to((ROOT/'kb').resolve()) or not support['quote']:
                raise ValueError('Support must come from the public KB')
            if support['quote'] not in source.read_text():
                raise ValueError('Support quote changed')
    return items, {'scope':SCOPE, 'proposal_sha256':runner.digest(path.read_bytes()),
                   'proposal_path':str(path.relative_to(ROOT)), 'independent_review':False,
                   'weak_labels':'parent_manual_design_before_predictions', 'count':len(items)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--proposal',type=Path,required=True)
    ap.add_argument('--name',required=True)
    ap.add_argument('--order',nargs='+',choices=sorted(runner.CANDIDATES),required=True)
    ap.add_argument('--resume',action='store_true')
    args = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.name) or len(set(args.order))!=len(args.order):
        ap.error('Use a simple name and unique candidate order')
    items, provenance = load_proposal(args.proposal)
    encoded = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items)
    load_environment()
    current = {**provenance, 'dataset_sha256':runner.digest(encoded.encode()),
        'protected':runner.protected_hashes(ROOT), 'pipelines':runner.pipeline_hashes(ROOT),
        'bootstrap_sha256':runner.digest(runner.BOOTSTRAP.encode()), 'order':args.order,
        'chat_model':os.environ['OPENAI_MODEL'], 'embedding_model':os.environ.get('EMBEDDING_MODEL')}
    dest = ROOT/'experiments/.cache'/('jh_manual_tutor_'+args.name)
    if args.resume:
        recorded=json.loads((dest/'plan.json').read_text())
        if any(recorded.get(k)!=v for k,v in current.items()):
            raise ValueError('Cannot resume with changed data, code or conditions')
    else:
        dest.mkdir()
        current['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        runner.write_json(dest/'plan.json',current)
        (dest/'inputs.jsonl').write_text(encoded)
    env=runner.server_env(os.environ,ROOT,'env')
    for label in args.order:
        run=dest/label
        if not (run/'runtime.json').exists():
            if run.exists():raise ValueError('Preserve partial collection and use a fresh run name')
            runner.collect_candidate(label,ROOT,items,run,env,32,'env',current)
        print('COLLECTED '+label,flush=True)
    summaries=[]
    for label in args.order:
        run=dest/label
        if not (run/'result.json').exists():
            rows=[json.loads(s) for s in (run/'responses.jsonl').read_text().splitlines()]
            runner.write_json(run/'result.json',score_rows(items,rows,32,journal=run/'judge.jsonl'))
        result=json.loads((run/'result.json').read_text())
        summary={'candidate':label, 'tutor':result['tutor'], 'invalid':result['invalid'],
                 'judge_errors':result['judge_errors'], 'runtime':json.loads((run/'runtime.json').read_text())}
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    if runner.protected_hashes(ROOT)!=current['protected'] or runner.pipeline_hashes(ROOT)!=current['pipelines']:
        raise ValueError('Code changed during experiment')
    runner.write_json(dest/'summary.json',summaries)
    (dest/'COMPLETE').write_text('Manual public KB development questions fully scored.\n')


if __name__=='__main__':main()
