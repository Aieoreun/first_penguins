"""Collect a fixed public-derived jamo stress diagnostic, never private data.

Noise is added only to normal/PII tutor questions. Their labels are inherited;
especially crying markers may change tone, so this is synthetic stress rather
than a hidden-set accuracy claim. HTTP payloads use the unchanged collector.
"""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from jh_compare import ROOT, load_public, run_variant
from pipelines.kimjaehak.config import VARIANTS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_items(source=None):
    if source is None:
        return load_public(endpoints=['tutor']), {
            'source': 'public/tutor.jsonl', 'source_sha256': digest(ROOT/'public/tutor.jsonl'),
            'source_kind': 'bundled_public_in_sample'}
    source = source.resolve()
    if not source.is_relative_to(ROOT/'experiments/.cache') or source.name != 'derived_inputs.jsonl':
        raise ValueError('Expected a local public-derived dataset')
    meta = json.loads((source.parent/'derivation.json').read_text())
    if (meta.get('scope') != 'public_paraphrase_full_diagnostic'
            or meta.get('dataset_sha256') != digest(source)):
        raise ValueError('Public paraphrase provenance differs')
    return list(map(json.loads, source.read_text().splitlines())), {
        'source': str(source.relative_to(ROOT)), 'source_sha256': digest(source),
        'source_kind': 'previously_derived_public_paraphrases', 'source_derivation': meta}


def perturb(items):
    output = []
    for item in items:
        if item['endpoint'] != 'tutor' or item['labels']['category'] not in {'normal', 'pii_disclosure'}:
            continue
        index = next((i for i in range(len(item['messages'])-1, -1, -1)
                      if item['messages'][i]['role'] == 'user'), None)
        if index is None:
            raise ValueError('Missing user message')
        for kind in ('prefix_laugh', 'suffix_cry'):
            row = copy.deepcopy(item)
            row.update(id=item['id']+'-noise-'+kind, source_id=item['id'], noise_kind=kind)
            text = row['messages'][index]['content']
            row['messages'][index]['content'] = 'ㅋㅋㅋ '+text if kind == 'prefix_laugh' else text+' ㅠㅠ'
            output.append(row)
    if len({row['id'] for row in output}) != len(output):
        raise ValueError('Duplicate source IDs')
    return output


def self_test():
    import unittest
    class Tests(unittest.TestCase):
        def test_only_noise_changes_and_originals_remain_untouched(self):
            item = {'id':'original', 'endpoint':'tutor', 'messages':[
                {'role':'user','content':'earlier'}, {'role':'assistant','content':'history'},
                {'role':'user','content':'question'}], 'labels':{'category':'normal','evidence':['S'],'gold_facts':['F']}}
            before = copy.deepcopy(item)
            rows = perturb([item])
            self.assertEqual(item, before)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row['labels'], item['labels'])
                self.assertEqual(row['messages'][:-1], item['messages'][:-1])
                self.assertEqual(row['source_id'], 'original')
            self.assertEqual(rows[0]['messages'][-1]['content'], 'ㅋㅋㅋ question')
            self.assertEqual(rows[1]['messages'][-1]['content'], 'question ㅠㅠ')

        def test_filters_are_dataset_only_and_duplicates_fail(self):
            item = {'id':'x','endpoint':'tutor','messages':[{'role':'user','content':'q'}],
                    'labels':{'category':'normal'}}
            self.assertEqual(perturb([{**item,'endpoint':'guard'}]), [])
            self.assertEqual(perturb([{**item,'labels':{'category':'emotional_distress'}}]), [])
            with self.assertRaises(ValueError):
                perturb([item,item])

        def test_provenance_rejects_nonpublic_source(self):
            with self.assertRaises(ValueError):
                source_items(Path('/tmp/untrusted.jsonl'))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--variant', choices=VARIANTS)
    ap.add_argument('--name')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--request-slots', type=int, default=64)
    ap.add_argument('--judge-workers', type=int, default=32)
    ap.add_argument('--embedding-mode', choices=('env','off'), default='env')
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if (not args.variant or not args.name or not re.fullmatch(r'[A-Za-z0-9_-]+', args.name)
            or not all(1 <= x <= 64 for x in (args.workers,args.request_slots,args.judge_workers))):
        ap.error('Invalid arguments')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=True)
    rows, source = source_items(args.source)
    items = perturb(rows)
    if not items:
        ap.error('No normal/PII tutor source items')
    root = ROOT/'experiments/.cache'/('jh_noise_'+args.name)
    root.mkdir(exist_ok=False)
    dataset = root/'derived_inputs.jsonl'
    dataset.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in items))
    provenance = {'scope':'public_noise_full_diagnostic', 'count':len(items), **source,
                  'dataset_sha256':digest(dataset), 'inherited_labels':True,
                  'independent_source_holdout':False, 'score_scope':'tutor',
                  'noise_policy':'fixed prefix_laugh and suffix_cry on normal/PII only',
                  'limitations':['Synthetic emotive markers may alter tone; no emotional recall claim.',
                                 'Labels and source IDs never enter runtime HTTP payloads.']}
    (root/'derivation.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    if args.build_only:
        print(json.dumps(provenance),flush=True)
        return
    args.endpoints, args.no_judge, args.collect_only = ['tutor'], False, True
    run_variant(args,args.variant,1,items,root)
    dest = root/(args.variant+'_1')
    manifest = json.loads((dest/'manifest.json').read_text())
    manifest.update(provenance)
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('DERIVED_SCORING_PENDING',dest,flush=True)


if __name__ == '__main__':
    main()
