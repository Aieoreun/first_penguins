"""Development grid for final-question retrieval; no independent holdout claim."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.kb import load_chunks
from core.search import Searcher
from pipelines.kimjaehak.query_gate import query_features, use_last


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--transfer', type=Path, required=True)
    ap.add_argument('--switch', type=Path, required=True)
    args = ap.parse_args()
    dest = args.output.resolve()
    if not dest.is_relative_to(ROOT/'experiments/.cache') or dest.exists():
        ap.error('Use a new local cache directory')
    sources = {'public': ROOT/'public/tutor.jsonl',
               'transfer': args.transfer.resolve(), 'switch': args.switch.resolve()}
    for name, path in sources.items():
        if name == 'public':
            continue
        if not path.is_relative_to(ROOT/'experiments/.cache'):
            ap.error('Expected local public-derived diagnostics')
        meta = json.loads((path.parent/'derivation.json').read_text())
        if (meta['dataset_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
                or meta['scope'] not in {'provided_kb_fact_diagnostic', 'public_paraphrase_full_diagnostic'}):
            ap.error('Unexpected diagnostic provenance')
    searcher = Searcher(load_chunks(ROOT/'kb'))
    data = {}
    for name, path in sources.items():
        rows = [r for r in map(json.loads, path.read_text().splitlines())
                if len(r['labels'].get('evidence') or []) == 1]
        data[name] = [dict(id=r['id'], **query_features(searcher, r['messages']),
                           gold=r['labels']['evidence']) for r in rows]
    reports = []
    for score, gap, terms in itertools.product((8,12,16,20,30,40), (0,.1,.2,.3), (4,8,12,20)):
        metrics = {}
        for name, rows in data.items():
            selected = [r['last'] if use_last(r, score, gap, terms) else r['base'] for r in rows]
            metrics[name] = {'count': len(rows), 'base_hits': sum(r['base'] in r['gold'] for r in rows),
                             'hits': sum(s in r['gold'] for s,r in zip(selected, rows)),
                             'changes': sum(s != r['base'] for s,r in zip(selected, rows))}
        reports.append({'min_score':score, 'min_gap':gap, 'min_terms':terms, 'metrics':metrics})
    result = {'scope':'query_gate_development_grid_not_holdout',
              'dataset_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()},
              'reports':reports, 'features':data}
    dest.mkdir()
    (dest/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'grids':len(reports), 'path':str(dest)}))


if __name__ == '__main__':
    main()
