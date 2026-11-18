"""Freeze independently reviewed KB contrasts and collect ordinary tutor HTTP.

Manual weak labels are development data, not a private evaluation or holdout.
Only the existing collector's messages-only payload goes to the server.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from jh_compare import run_variant
from pipelines.kimjaehak.config import VARIANTS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_file(path, parent):
    path = Path(path).resolve()
    if not path.is_relative_to(parent.resolve()) or not path.is_file():
        raise ValueError('Missing file or path outside expected root')
    return path


def build_items(proposal_path, review_path, root=ROOT):
    root = Path(root).resolve()
    cache = root/'experiments/.cache'
    proposal_path, review_path = (local_file(p, cache) for p in (proposal_path, review_path))
    proposal, review = (json.loads(p.read_text()) for p in (proposal_path, review_path))
    design = proposal.get('design_constraints', {})
    if (proposal.get('scope') != 'manual_organizer_kb_contrast_development_not_private_not_holdout'
            or review.get('scope') != 'independent_manual_kb_contrast_review'
            or review.get('proposal_sha256') != sha(proposal_path)
            or review.get('kb_manifest_sha256') != proposal.get('kb_manifest_sha256')
            or review.get('model_predictions_inspected') is not False or review.get('api_calls') != 0
            or design.get('model_predictions_not_inspected_during_design') is not True
            or design.get('pipeline_not_evaluated') is not True or design.get('api_calls') != 0):
        raise ValueError('Unverified or changed independent review')
    entries = proposal['kb_files']
    paths = [row['path'] for row in entries]
    actual = {str(p.relative_to(root)) for p in (root/'kb').rglob('*') if p.is_file()}
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    if (len(paths) != len(set(paths)) or set(paths) != actual or paths != sorted(paths)
            or hashlib.sha256(canonical).hexdigest() != proposal['kb_manifest_sha256']):
        raise ValueError('KB file inventory differs')
    for row in entries:
        path = local_file(root/row['path'], root/'kb')
        if (set(row) != {'path', 'sha256', 'bytes'} or sha(path) != row['sha256']
                or len(path.read_bytes()) != row['bytes']):
            raise ValueError('KB changed since question design')
    pairs = proposal['pairs']
    ids = [pair['id'] for pair in pairs]
    accepted, excluded = review['accepted_pair_ids'], [r['id'] for r in review['excluded']]
    if (len(ids) != len(set(ids)) or not accepted
            or len(accepted + excluded) != len(set(accepted + excluded))
            or set(accepted + excluded) != set(ids)):
        raise ValueError('Review must account for every pair exactly once')
    items, questions = [], set()
    for pair in pairs:
        pair_data = json.dumps([pair['supported']['question'], pair['unsupported']['question']],
                               ensure_ascii=False, separators=(',', ':')).encode()
        if hashlib.sha256(pair_data).hexdigest() != pair['question_pair_sha256']:
            raise ValueError('Question pair hash differs')
        if pair['id'] not in accepted:
            continue
        pos = pair['supported']
        support = local_file(root/pos['source_path'], root/'kb')
        quote = pos['support_quote']
        section = pair['source_section']
        if (not isinstance(section, str) or not re.fullmatch(r'[A-Z]{2,4}-\d{2}-\d{2}', section)
                or not re.fullmatch(re.escape(section)+r'(?:-table\d+)?\.(?:md|csv)', support.name)):
            raise ValueError('Positive source section differs from source file')
        if (sha(support) != pos['source_sha256'] or not isinstance(quote, str) or not quote
                or quote not in support.read_text()
                or hashlib.sha256(quote.encode()).hexdigest() != pos['quote_sha256']):
            raise ValueError('Positive support quote differs')
        extra = pos.get('additional_support_quotes', [])
        if (not isinstance(extra, list) or any(not isinstance(q, str) or not q or q not in support.read_text()
                                             for q in extra)):
            raise ValueError('Additional support quote differs')
        for kind, category, action in (('supported', 'normal', 'answer'),
                                       ('unsupported', 'out_of_material', 'abstain')):
            row = pair[kind]
            question, labels = row['question'], row['labels']
            facts = labels.get('gold_facts') if isinstance(labels, dict) else None
            if not isinstance(facts, list) or any(not isinstance(f, str) or not f.strip() for f in facts):
                raise ValueError('Expected a list of nonempty gold fact strings')
            if (not isinstance(question, str) or not question.strip() or question in questions
                    or labels.get('category') != category or labels.get('action') != action
                    or (kind == 'supported' and (labels.get('evidence') != [pair['source_section']]
                                                or not labels.get('gold_facts')))
                    or (kind == 'unsupported' and (labels.get('evidence') != [] or labels.get('gold_facts') != []))):
                raise ValueError('Invalid or duplicate reviewed contrast')
            questions.add(question)
            item = {'id': pair['id']+'-'+kind, 'source_id': pair['id'], 'endpoint': 'tutor',
                    'messages': [{'role': 'user', 'content': question}], 'labels': deepcopy(labels),
                    'contrast_role': kind, 'domain': pair['domain']}
            if kind == 'supported':
                item['label_support_quote'] = quote
            items.append(item)
    metadata = {'scope': 'provided_kb_fact_diagnostic', 'diagnostic_kind': 'manual_kb_contrast_v1',
                'source': str(proposal_path.relative_to(root)), 'source_sha256': sha(proposal_path),
                'review': str(review_path.relative_to(root)), 'review_sha256': sha(review_path),
                'kb_manifest_sha256': proposal['kb_manifest_sha256'],
                'weak_labels': 'manual_KB_design_then_independent_review_before_model_inspection',
                'generation_model': None, 'independent_source_holdout': False,
                'public_questions_in_generation': False, 'accepted_pair_ids': accepted,
                'score_interpretation': 'Tutor-only balanced stress diagnostic, not private-score estimate'}
    return items, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--request-slots', type=int, default=64)
    parser.add_argument('--judge-workers', type=int, default=32)
    parser.add_argument('--embedding-mode', choices=('env', 'off'), default='env')
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    if (not re.fullmatch(r'[A-Za-z0-9_-]+', args.name)
            or any(not 1 <= n <= 64 for n in (args.workers, args.request_slots, args.judge_workers))):
        parser.error('Invalid name or concurrency')
    items, provenance = build_items(args.proposal, args.review)
    dest = ROOT/'experiments/.cache'/('jh_kb_contrast_'+args.name)
    dest.mkdir(exist_ok=False)
    dataset = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in items)
    provenance.update(count=len(items), dataset_sha256=hashlib.sha256(dataset.encode()).hexdigest())
    (dest/'derived_inputs.jsonl').write_text(dataset)
    (dest/'derivation.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    if args.build_only:
        print(json.dumps({'count':len(items), 'dataset_sha256':provenance['dataset_sha256']}))
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=True)
    args.no_judge, args.collect_only, args.endpoints = False, True, ['tutor']
    run_variant(args, args.variant, 1, items, dest)
    run = dest/(args.variant+'_1')
    manifest = json.loads((run/'manifest.json').read_text())
    manifest.update(provenance)
    (run/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print('DERIVED_SCORING_PENDING', run)


if __name__ == '__main__':
    main()
