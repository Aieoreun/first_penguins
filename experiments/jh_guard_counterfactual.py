"""Offline, manually reviewed public/KB-derived guard grounding contrasts.

Development diagnostic only: never training or deployment data, never an
independent holdout. This module makes no model or HTTP calls.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SHA = '4a585c174f15dec205a286ec45e8c14ff28b01bbcedde3097072ca52e3b0c426'
KB_SHA = '8b952ddc320d0f21c2eefc8231b4856986e79e5f94ea44303b79997972e8fb2e'
DESTINATION = 'experiments/.cache/jh_guard_counterfactual_v1'

# Manually source-reviewed before candidate predictions; parent review pending.
PROPOSALS = [
    {'source_id': 'pub-78410d6c', 'kb_file': 'kb/finance/FIN-04-01.md',
     'support_quote': '다람쥐 틴즈 체크카드는 만 12세 이상이 사용할 수 있어요.',
     'old': '12', 'new': '11', 'unsupported': '만 11세 이상이 사용할 수 있어요.',
     'reason': '최소 나이보다 어린 11세까지 허용하므로 교재의 고정 조건과 모순된다.'},
    {'source_id': 'pub-5333d544', 'kb_file': 'kb/finance/FIN-03-02.md',
     'support_quote': '다람쥐 틴즈 체크카드는 밤 10시부터 오전 6시까지 결제가 막혀요.',
     'old': '오전 6시', 'new': '오전 7시', 'unsupported': '밤 10시부터 오전 7시까지 결제가 막혀 있어요.',
     'reason': '오전 5시 결제 불가 결론은 유지되지만 차단 종료 시각을 잘못 연장한다.'},
    {'source_id': 'pub-946de9f8', 'kb_file': 'kb/finance/FIN-02-02.md',
     'support_quote': '우대 금리는 12번을 빠짐없이 넣는 조건을 채워야 더해져요. 한 번을 빠뜨리면 이 조건을 충족하지 못해요.',
     'old': '12번', 'new': '13번', 'unsupported': '13번을 빠짐없이 넣어야 하는 우대 조건',
     'reason': '상품에 정해진 우대 금리 납입 횟수를 바꾼다. 기존 여러 턴의 대화는 보존한다.'},
    {'source_id': 'pub-7cba6fa3', 'kb_file': 'kb/python/PY-01-01.md',
     'support_quote': '한 실습은 하루에 10번까지 제출할 수 있어요.',
     'old': '10번', 'new': '11번', 'unsupported': '하루에 11번까지 제출할 수 있어요.',
     'reason': '가변 코드 입력이 아니라 과정의 고정된 하루 제출 상한을 바꾼다.'},
    {'source_id': 'pub-f15ab968', 'kb_file': 'kb/finance/FIN-01-04.md',
     'support_quote': '하루 동안 다른 통장으로 보내는 돈을 합해 5만 원까지 이체할 수 있다는 뜻이에요.',
     'old': '5만 원', 'new': '4만 원', 'unsupported': '하루 이체 한도는 4만 원',
     'reason': '6만 원 이체 불가 결론은 유지되지만 상품의 정확한 고정 한도를 바꾼다.'},
    {'source_id': 'pub-5d501bc6', 'kb_file': 'kb/python/PY-02-01.md',
     'support_quote': '이 과정의 학점 예제는 점수를 0점부터 100점까지로 정해요.',
     'old': '100점', 'new': '120점', 'unsupported': '0점부터 120점까지',
     'reason': '임의의 코드 입력값이 아니라 이 과정에서 정한 허용 점수 범위를 바꾼다.'},
    {'source_id': 'pub-242ef221', 'kb_file': 'kb/python/PY-03-04.md',
     'support_quote': '60점 이상 80점 미만이면 B0, 40점 이상 60점 미만이면 C0',
     'old': '60점', 'new': '50점', 'unsupported': '평균이 50점 이상 80점 미만이면 B0를 출력해요.',
     'reason': '50점 이상 60점 미만은 C0이므로 하한을 내리면 명백히 잘못 분류한다.'},
    {'source_id': 'pub-995e13b6', 'kb_file': 'kb/finance/FIN-03-04-table1.csv',
     'support_quote': '다람 문구점은 환불할 때 원래 결제 수단으로 돈을 돌려줘요.',
     'old': '원래 결제 수단', 'new': '현금', 'unsupported': '환불할 때 현금으로 돈을 돌려줘요.',
     'reason': '특정 현금 구매를 가정하지 않은 가게의 일반 규칙을 현금 환불로 바꿔 교재와 모순된다.'},
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def kb_inventory(root):
    root = Path(root).resolve()
    files = {}
    for path in sorted((root / 'kb').rglob('*')):
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(root / 'kb'):
                raise ValueError('KB file escapes source tree')
            files[str(path.relative_to(root))] = digest(path.read_bytes())
    return files


def build_items(root, proposals=PROPOSALS, public_sha=PUBLIC_SHA, kb_sha=KB_SHA, expected_pairs=8):
    """Verify all inputs before making exactly one substitution per wrong answer."""
    root = Path(root).resolve()
    source = root / 'public/guard.jsonl'
    if source.is_symlink() or digest(source.read_bytes()) != public_sha:
        raise ValueError('Public guard source changed')
    inventory = kb_inventory(root)
    if digest(canonical(inventory)) != kb_sha:
        raise ValueError('KB inventory/content changed')
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    parents = {row['id']: row for row in records}
    if len(parents) != len(records):
        raise ValueError('Repeated public IDs')
    if len(proposals) != expected_pairs or len({p['source_id'] for p in proposals}) != expected_pairs:
        raise ValueError('Unexpected pair count or repeated source')
    items, provenance = [], []
    for proposal in proposals:
        parent = parents.get(proposal['source_id'])
        labels = parent.get('labels', {}) if parent else {}
        if (parent is None or parent.get('endpoint') != 'guard'
                or labels.get('category') != 'normal' or labels.get('decision') != 'allow'
                or labels.get('grounded') is not True or labels.get('safe') is not True
                or labels.get('unsupported') != []):
            raise ValueError('Parent must be normal/allow and safely grounded')
        answer = parent.get('candidate_answer')
        old, new, span = (proposal.get(key) for key in ('old', 'new', 'unsupported'))
        if (not isinstance(answer, str) or any(not isinstance(value, str) or not value for value in (old, new, span))
                or old == new or answer.count(old) != 1 or new in answer):
            raise ValueError('Substitution must change exactly one unambiguous occurrence')
        filename, quote = proposal.get('kb_file'), proposal.get('support_quote')
        if (filename not in inventory or not isinstance(quote, str) or not quote
                or quote not in (root / filename).read_text()
                or not isinstance(proposal.get('reason'), str) or not proposal['reason'].strip()):
            raise ValueError('Missing exact KB quotation or manual reason')
        wrong_answer = answer.replace(old, new, 1)
        changed_start = answer.index(old)
        if wrong_answer.count(span) != 1:
            raise ValueError('Unsupported span must uniquely match the actual wrong answer')
        span_start = wrong_answer.index(span)
        if not (span_start <= changed_start and span_start + len(span) >= changed_start + len(new)):
            raise ValueError('Unsupported span does not cover the substituted text')
        for kind, grounded in (('control', True), ('counterfactual', False)):
            item = copy.deepcopy(parent)
            item.update(id=parent['id'] + '/' + kind, source_id=parent['id'], pair_id=parent['id'], contrast_kind=kind)
            if not grounded:
                item['candidate_answer'] = wrong_answer
                item['labels'].update(grounded=False, unsupported=[span])
            items.append(item)
        provenance.append({**copy.deepcopy(proposal), 'kb_sha256': inventory[filename],
                           'original_answer_sha256': digest(answer.encode()),
                           'counterfactual_answer_sha256': digest(wrong_answer.encode()),
                           'replacement_start': changed_start, 'unsupported_start': span_start})
    manifest = {
        'scope': 'public_guard_counterfactual16_manual_grounding_development',
        'count': len(items), 'pairs': expected_pairs, 'candidate_count': len(items),
        'source': 'public/guard.jsonl', 'source_sha256': public_sha,
        'kb_files': inventory, 'kb_manifest_sha256': kb_sha,
        'kb_manifest_hash_formula': 'sha256(json.dumps(kb_files, ensure_ascii=False, sort_keys=True, separators=(comma,colon)).encode(utf-8))',
        'proposal_sha256': digest(canonical(proposals)), 'pairs_provenance': provenance,
        'generation_api_calls': 0, 'external_sources_used': False,
        'model_predictions_inspected_during_design': False,
        'independent_source_holdout': False, 'inherited_public_parents': True,
        'training_allowed': False, 'deployment_allowed': False,
        'review_status': 'pending_parent_review_before_scoring',
        'primary_metrics': ['control_accuracy', 'counterfactual_detection', 'paired_both_correct'],
        'interpretation': 'Manual KB-supported counterfactual development data. Grounded-label accuracy is primary; official span score is supplementary because manual span boundaries are subjective. Public parents may be fitted; this is not private or independent holdout evidence.',
    }
    return items, manifest


def write_dataset(root=ROOT):
    root = Path(root).resolve()
    items, manifest = build_items(root)
    data = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items).encode()
    manifest['dataset_sha256'] = digest(data)
    destination = root / DESTINATION
    if not destination.resolve().is_relative_to(root / 'experiments/.cache'):
        raise ValueError('Output must remain in ignored experiment cache')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'inputs.jsonl').write_bytes(data)
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return destination, manifest


def grounding_metrics(items, rows):
    """Grounding only, after official shape validation; invalid/missing is wrong."""
    from grader.scoring import parse_guard
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate response IDs')
    by_id = {row['id']: row for row in rows}
    if len({item['id'] for item in items}) != len(items) or set(by_id) - {item['id'] for item in items}:
        raise ValueError('Unexpected or duplicate item IDs')
    pairs = {}
    invalid = 0
    for item in items:
        kind = item.get('contrast_kind')
        if kind not in ('control', 'counterfactual') or type(item['labels'].get('grounded')) is not bool:
            raise ValueError('Missing contrast grounding contract')
        if item['labels']['grounded'] is not (kind == 'control'):
            raise ValueError('Contrast label differs from pair role')
        pair = pairs.setdefault(item['pair_id'], {})
        if kind in pair:
            raise ValueError('Repeated contrast role')
        row = by_id.get(item['id'], {})
        parsed = None if row.get('error') else parse_guard(row.get('response'), True)
        invalid += parsed is None
        pair[kind] = parsed is not None and parsed['output']['grounded'] is item['labels']['grounded']
    if not pairs or any(set(pair) != {'control', 'counterfactual'} for pair in pairs.values()):
        raise ValueError('Incomplete contrast pairs')
    count = len(pairs)
    return {'pairs': count, 'count': len(items), 'invalid': invalid,
            'control_accuracy': sum(pair['control'] for pair in pairs.values()) / count,
            'counterfactual_detection': sum(pair['counterfactual'] for pair in pairs.values()) / count,
            'paired_both_correct': sum(all(pair.values()) for pair in pairs.values()) / count}


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    path, manifest = write_dataset()
    print(json.dumps({'path': str(path), 'count': manifest['count'], 'dataset_sha256': manifest['dataset_sha256']}))
