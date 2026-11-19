"""Refresh continuous tutor experiment checkpoints without any API calls."""
import hashlib
import json

from jh_compare import ROOT
from jh_round2_summary import summarize


def main():
    cache=ROOT/'experiments/.cache'
    report={'scope':'tutor_continuous_development','stop_remaining_percent':60,
            'generation':[],'draft_selection':[],'ranker':[],'pending':[]}
    report['refusal_generation'] = []
    for pattern,key in [('jh_generation_r3_*/manifest.json','generation'),('jh_draft_r3_*/manifest.json','draft_selection'),
                        ('jh_refusal_generation_r3_*/manifest.json','refusal_generation')]:
        for meta in sorted(cache.glob(pattern)):
            p=meta.parent/'result.json'
            if not p.exists():
                report['pending'].append({'run':str(meta.parent.relative_to(ROOT)),'phase':'generation_or_scoring'})
                continue
            r=json.loads(p.read_text())
            fields=('scope','variant','method','count','source','source_selection','commit','baseline_tutor','revised_tutor',
                    'first_valid_tutor','source_first_valid_tutor','tutor','attempted','error_requests',
                    'compared','changed_from_selected','invalid','judge_errors','scoring_status')
            entry={k:r[k] for k in fields if k in r}
            entry.update(result_path=str(p.relative_to(ROOT)),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            report[key].append(entry)
    for p in sorted(cache.glob('jh_ranker_r3_*/result.json')):
        r=json.loads(p.read_text())
        entry={k:v for k,v in r.items() if k!='reports'}
        entry['reports']={k:{dataset:{f:v for f,v in metric.items() if f!='predictions'}
                              for dataset,metric in results.items()} for k,results in r['reports'].items()}
        entry.update(result_path=str(p.relative_to(ROOT)),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        report['ranker'].append(entry)
    http=summarize('*r3*/*/manifest.json')
    report['http']=http['records']
    report['pending'].extend(http['pending'])
    report['failed']=http['failed']
    report['material_votes'] = []
    for meta in sorted(cache.glob('jh_material_vote_probe_r3_*/manifest.json')):
        result_path = meta.parent/'result.json'
        if not result_path.exists():
            report['pending'].append({'run':str(meta.parent.relative_to(ROOT)), 'phase':'material_component'})
            continue
        report['material_votes'].append({**json.loads(result_path.read_text()),
            'result_path':str(result_path.relative_to(ROOT)),
            'result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256(meta.read_bytes()).hexdigest()})
    report['kb_generation'] = []
    for meta in sorted(cache.glob('jh_kb_facts_r3_*/manifest.json')):
        diagnostic = meta.parent / 'diagnostic.json'
        if not diagnostic.exists():
            report['pending'].append({'run':str(meta.parent.relative_to(ROOT)), 'phase':'kb_generation'})
            continue
        manifest = json.loads(meta.read_text())
        entry = {**json.loads(diagnostic.read_text()),
                 'run':str(meta.parent.relative_to(ROOT)),
                 'question_style':manifest.get('question_style', 'fact'),
                 'model':manifest['model'], 'kb_sha256':manifest['kb_sha256'],
                 'prompt_sha256':manifest['prompt_sha256'],
                 'manifest_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),
                 'diagnostic_sha256':hashlib.sha256(diagnostic.read_bytes()).hexdigest()}
        review = cache / meta.parent.name.replace('jh_kb_facts_', 'jh_kb_review_', 1) / 'reviews.jsonl'
        if review.exists():
            rows = [json.loads(line) for line in review.read_text().splitlines()]
            entry['review'] = {'count':len(rows), 'accepted':sum(r['valid'] for r in rows),
                               'errors':sum(bool(r['error']) for r in rows),
                               'sha256':hashlib.sha256(review.read_bytes()).hexdigest()}
        report['kb_generation'].append(entry)
    (ROOT/'docs/jh/tutor_continuous_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:len(v) for k,v in report.items() if isinstance(v,list)}))


if __name__=='__main__':main()
