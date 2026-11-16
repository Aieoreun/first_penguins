"""Summarize resumed tutor experiments without new model or judge calls."""
import hashlib
import json

from jh_compare import ROOT
from jh_round2_summary import summarize


def main():
    http = summarize('*r4*/*/manifest.json')
    report = {'scope': 'round4_tutor_development_not_private_evaluation',
              'baseline': 'jh_r3_selected_focused', 'stop_remaining_percent': 40,
              'http': http['records'], 'pending': http['pending'], 'failed': http['failed'],
              'components': []}
    cache = ROOT / 'experiments/.cache'
    for pattern in ('jh_generation_r4_*/manifest.json', 'jh_refusal_generation_r4_*/manifest.json'):
        for meta in sorted(cache.glob(pattern)):
            result = meta.parent / 'result.json'
            if not result.exists():
                report['pending'].append({'run': str(meta.parent.relative_to(ROOT)),
                                          'phase': 'component_collection_or_scoring'})
                continue
            data = json.loads(result.read_text())
            entry = {key: data[key] for key in ('scope', 'variant', 'source', 'count', 'commit',
                     'baseline_tutor', 'revised_tutor', 'attempted', 'error_requests', 'scoring_status')
                     if key in data}
            entry.update(result_path=str(result.relative_to(ROOT)),
                         result_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
            report['components'].append(entry)
    (ROOT / 'docs/jh/round4_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: len(value) for key, value in report.items() if isinstance(value, list)}))


if __name__ == '__main__':
    main()
