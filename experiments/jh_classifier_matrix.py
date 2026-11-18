"""Frozen grouped CV for the 2x2 representation/classifier comparison. API zero."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from pipelines.kimjaehak.classifier_models import CategoryModel, KINDS, public_data

BLOCKED = {'harmful_content', 'academic_dishonesty', 'jailbreak_injection', 'off_topic'}
SEEDS = (20261118, 20261119, 20261120)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_folds(texts, labels):
    # Duplicate detection only; classifier TF-IDF is fitted anew on each train fold.
    near = TfidfVectorizer(analyzer='char', ngram_range=(3, 5)).fit_transform(texts)
    sim = (near @ near.T).toarray()
    parent = list(range(len(texts)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(texts)):
        for j in range(i):
            if sim[i, j] >= .82:
                parent[find(i)] = find(j)
    groups = np.array([find(i) for i in range(len(texts))])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=20261116).split(texts, labels, groups))
    for a, b in folds:
        if set(groups[a]) & set(groups[b]):
            raise ValueError('Group overlap')
    return groups, folds


def metrics(gold, prediction):
    gold, prediction = np.asarray(gold), np.asarray(prediction)
    gb = np.array([x in BLOCKED for x in gold]); pb = np.array([x in BLOCKED for x in prediction])
    mask = gold != 'normal'
    return {'accuracy': float(np.mean(gold == prediction)),
            'macro_f1': float(f1_score(gold, prediction, average='macro', zero_division=0)),
            'block_f1': float(f1_score(gb, pb, zero_division=0)),
            'category_non_normal': float(np.mean(gold[mask] == prediction[mask])) if mask.any() else None,
            'false_block': int(np.sum(~gb & pb)), 'missed_block': int(np.sum(gb & ~pb))}


def selective(gold, pred, conf):
    # Harmful subtype stays a separate common model at deployment.
    eligible = np.flatnonzero(pred != 'harmful_content')
    ordered = eligible[np.argsort(-conf[eligible], kind='stable')]
    result = {}
    for fraction in (.10, .25, .50, .75):
        selected = ordered[:min(len(ordered), round(len(gold) * fraction))]
        result[str(fraction)] = {'count': len(selected),
            'category_errors': int(np.sum(gold[selected] != pred[selected])),
            'accuracy': float(np.mean(gold[selected] == pred[selected])) if len(selected) else None,
            'min_confidence': float(conf[selected].min()) if len(selected) else None}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not args.name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid name')
    dest = ROOT / 'experiments/.cache' / ('jh_classifier_matrix_' + args.name)
    dest.mkdir(exist_ok=False)
    rows, texts, dense, embedding_model = public_data()
    labels = np.array([r['labels']['category'] for r in rows])
    groups, folds = group_folds(texts, labels)
    sources = ['public/tutor.jsonl', 'public/guard.jsonl',
        'pipelines/kimjaehak/assets/public_category_embeddings.npz',
        'pipelines/kimjaehak/assets/public_category_embeddings.json',
        'pipelines/kimjaehak/classifier_models.py', 'experiments/jh_classifier_matrix.py']
    manifest = {'scope': 'public_grouped_oof_classifier_matrix_not_private_score',
        'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'sources': {p: digest(ROOT / p) for p in sources}, 'embedding_model': embedding_model,
        'ids': [r['id'] for r in rows], 'text_mode': 'last_user', 'group_count': len(set(groups)),
        'folds': [{'train': a.tolist(), 'test': b.tolist()} for a,b in folds],
        'config': {'width':32,'alpha':.1,'C':3000,'seeds':SEEDS,'solver':'lbfgs','max_iter':600},
        'api_calls': 0, 'independent_holdout': False,
        'note': 'Fixed architecture screen. Coverage curves are descriptive OOF ranks, not independently calibrated deployment thresholds.'}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    reports = []
    for kind in KINDS:
        for seed in (SEEDS if kind.endswith('mlp') else SEEDS[:1]):
            prediction = np.empty(len(rows), dtype=object); confidence = np.zeros(len(rows))
            diagnostics = []
            for fold, (a,b) in enumerate(folds):
                model = CategoryModel(kind, seed).fit([texts[i] for i in a], labels[a], dense[a])
                p = model.probabilities([texts[i] for i in b], dense[b])
                prediction[b], confidence[b] = model.classes[p.argmax(axis=1)], p.max(axis=1)
                diagnostics.append(model.diagnostics)
                print(json.dumps({'kind':kind,'seed':seed,'fold':fold,'fit':model.diagnostics}), flush=True)
            report = {'kind':kind,'seed':seed,'metrics':metrics(labels,prediction),
                      'matched_coverage':selective(labels,prediction,confidence),'fold_diagnostics':diagnostics}
            reports.append(report)
            (dest / f'{kind}_{seed}_predictions.json').write_text(json.dumps([
                {'id':r['id'],'gold':str(labels[i]),'prediction':str(prediction[i]),'confidence':float(confidence[i])}
                for i,r in enumerate(rows)], indent=2)+'\n')
            (dest / 'results.json').write_text(json.dumps(reports,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k != 'fold_diagnostics'}), flush=True)
    if any(digest(ROOT/p) != sha for p,sha in manifest['sources'].items()):
        raise ValueError('Inputs changed during CV')
    (dest / 'COMPLETE').write_text('All fixed grouped folds completed; API zero.\n')


if __name__ == '__main__':
    main()
