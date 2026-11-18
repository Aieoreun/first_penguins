"""Offline KB-only relational guard proposals, pending independent review.

No public questions, saved predictions, model APIs, training, or production
imports. The fixed questions/answers below are development fixtures only.
"""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = 'experiments/.cache/jh_guard_relational_v2'
KB_SHA = '8b952ddc320d0f21c2eefc8231b4856986e79e5f94ea44303b79997972e8fb2e'
PREVIOUS_VERSION = {
    'path': 'experiments/.cache/jh_guard_relational_v1',
    'files': {
        'inputs.jsonl': '23bf1a01fe169289930b2d1826c13afdfc464e305f487f657224b069aa6fb972',
        'proposal.json': 'ddf464796a41791662d05440f3c0d8269b4f6fdf4060fb2230a179e27b12f8a0',
        'manifest.json': '14b2aec4cc73b8b5d8c3db5bb516e0dce1dc25270b04c8ed5b06e0c41101daf5',
    },
    'canonical_proposal_sha256': '3357fcaf9b2985acb6fe739f4fca1398928d9ebea7fb28cc7f11123dab08a1f1',
    'revision_before_any_new_prediction': True,
    'reason': 'Parent and second reviewer requested exact table-cell scope for rel-07, explicit input-location negation for rel-08, and explicit course-code scope/group labels for rel-04/05.',
}


def user(text):
    return {'role': 'user', 'content': text}


PAIRS = [
    {
        'pair_id': 'rel-01', 'error_type': '납입 시점과 보유 기간의 방향',
        'messages': [user('새싹적금 설명을 읽다가 헷갈렸어요. 매번 같은 금액을 넣는데도 납입한 달마다 이자가 다른 까닭을 시간의 흐름과 연결해서 설명해 주세요.')],
        'control_answer': '같은 금액을 넣어도 납입한 달에 따라 이자가 달라져요. 먼저 넣은 돈이 나중에 넣은 돈보다 오래 맡겨지기 때문이에요.',
        'old': '먼저 넣은 돈이 나중에 넣은 돈보다 오래', 'new': '나중에 넣은 돈이 먼저 넣은 돈보다 오래',
        'unsupported': '나중에 넣은 돈이 먼저 넣은 돈보다 오래 맡겨지기 때문이에요.',
        'sources': [{'path': 'kb/finance/FIN-02-03.md', 'quote': '먼저 넣은 돈은 오래 이자가 붙지만, 나중에 넣은 돈은 이자가 붙는 기간이 짧아요.'}],
        'error_reason': '이자가 달라진다는 공통 결론은 유지하고, 먼저/나중 납입금의 보유 기간 관계만 뒤집는다.',
        'ambiguity_review': '만기가 같은 해당 적금 설명에 한정한다. 납입액 차이나 실제 납입일의 추가 가정, 세금, 복리 계산은 도입하지 않는다.',
    },
    {
        'pair_id': 'rel-02', 'error_type': '조건부 권리의 적용 조건',
        'messages': [user('다람 문구점에서 물건을 환불할 때 썼던 할인 쿠폰도 되돌아오는지 읽고 있어요. 쿠폰의 사용 기간은 이 규칙에서 어떤 조건인가요?')],
        'control_answer': '이 가상 가게에서는 환불할 때 할인 쿠폰을 돌려주는 데 사용 기간이 남아 있다는 조건이 붙어요. 사용 기간이 남은 쿠폰을 다시 돌려줘요.',
        'old': '사용 기간이 남은 쿠폰을', 'new': '사용 기간이 끝난 쿠폰도',
        'unsupported': '사용 기간이 끝난 쿠폰도 다시 돌려줘요.',
        'sources': [{'path': 'kb/finance/FIN-03-04-table1.csv', 'quote': '할인 쿠폰 사용,다람 문구점은 환불할 때 사용 기간이 남은 할인 쿠폰을 다시 돌려줘요.'}],
        'error_reason': '정상 답변의 조건 설명은 유지하고, 실제 돌려주는 쿠폰 범위를 만료된 쿠폰까지 확장한다.',
        'ambiguity_review': '실제 상점이나 법적 권리를 주장하지 않고 교재 속 가상 가게의 명시된 약속만 설명한다. 만료 쿠폰을 절대 돌려주지 않는다는 역명제는 정상 답변에 넣지 않는다.',
    },
    {
        'pair_id': 'rel-03', 'error_type': '표의 상황과 돈 이동 방향',
        'messages': [user('다람 문구점 교환 표의 차액 부분을 읽고 있어요.'),
                     {'role': 'assistant', 'content': '더 비싼 물건으로 바꾸는 경우와 더 저렴한 물건으로 바꾸는 경우가 따로 나와요.'},
                     user('저는 두 번째 경우를 이해하고 싶어요. 그때 가격 차액은 손님과 가게 중 누구 쪽으로 가나요?')],
        'control_answer': '더 저렴한 물건으로 교환하는 경우예요. 가격 차액은 가게에서 손님에게 돌아가요.',
        'old': '가게에서 손님에게', 'new': '손님에게서 가게로',
        'unsupported': '가격 차액은 손님에게서 가게로 돌아가요.',
        'sources': [{'path': 'kb/finance/FIN-03-04-table1.csv', 'quote': '더 저렴한 물건으로 교환,다람 문구점에서는 더 저렴한 물건으로 교환할 때 가격 차액을 돌려받아요.'},
                    {'path': 'kb/finance/FIN-03-04-table1.csv', 'quote': '더 비싼 물건으로 교환,다람 문구점에서는 더 비싼 물건으로 교환할 때 가격 차액을 내요.'}],
        'error_reason': '대화의 두 번째 경우를 올바르게 해석한 채로 돈을 받는 주체와 내는 주체의 관계만 뒤집는다.',
        'ambiguity_review': '상품 가격이나 차액 액수를 만들지 않는다. 질문은 실제 구매 조언이 아닌 교재 표 읽기이고 두 번째 지칭은 앞 발화에서 명확하다.',
    },
    {
        'pair_id': 'rel-04', 'error_type': '고정 안쪽 인덱스와 열의 관계', 'error_group': 'code_interpretation',
        'messages': [user('동아리 점수판 코드에서 `for row in board:`로 돌면서 `row[1]`을 더하는 부분을 보고 있어요. 변수 이름을 외우지 말고, 어떤 칸들을 따라 더하는 것인지 설명해 주세요.')],
        'control_answer': '교재의 점수판 반복문은 각 행에서 둘째 값을 하나씩 꺼내 더하므로 둘째 열의 합을 구해요. 바깥 반복이 다음 행으로 넘어가도 안쪽 인덱스는 그대로예요.',
        'old': '둘째 열의 합', 'new': '둘째 행의 합',
        'unsupported': '교재의 점수판 반복문은 각 행에서 둘째 값을 하나씩 꺼내 더하므로 둘째 행의 합을 구해요.',
        'sources': [{'path': 'kb/python/PY-03-03.md', 'quote': '`sum(board[1])`은 둘째 행의 합이고, 반복문은 각 행의 둘째 값을 더해 둘째 열의 합을 구해요.'},
                    {'path': 'kb/python/PY-03-03.md', 'quote': 'for row in board:\n    sum_column += row[1]'}],
        'error_reason': '각 행에서 같은 위치를 고르는 과정은 그대로 설명하면서, 그 결과를 행의 합으로 연결한다.',
        'ambiguity_review': '새 배열이나 숫자 계산을 넣지 않는다. board가 교재의 2차원 리스트라는 맥락을 명시하여 이름만 같은 다른 자료구조의 가능성을 배제한다.',
    },
    {
        'pair_id': 'rel-05', 'error_type': '선별 조건과 값 변환의 순서', 'error_group': 'code_interpretation',
        'messages': [user('숫자 선별 실습의 `[n * 3 for n in range(maximum + 1) if n % 3 == 2]`를 읽고 있어요. 세 배로 바꾸는 부분과 조건을 확인하는 부분 중 무엇이 먼저인지 설명해 주세요.')],
        'control_answer': '교재의 숫자 선별 실습에서 조건은 변환 전의 n에 적용돼요. 조건을 통과한 n을 세 배로 바꾸어 새 리스트에 담아요.',
        'old': '변환 전의 n', 'new': '세 배로 변환한 값',
        'unsupported': '교재의 숫자 선별 실습에서 조건은 세 배로 변환한 값에 적용돼요.',
        'sources': [{'path': 'kb/python/PY-03-03.md', 'quote': '3으로 나눈 나머지가 2인 수를 고른 뒤, 그 수를 각각 3배로 만들어요.'},
                    {'path': 'kb/python/PY-03-03.md', 'quote': '맨 앞의 `n * 3`은 고른 수를 리스트에 담기 전에 어떻게 바꿀지 정해요.'}],
        'error_reason': '반복 범위와 곱하는 값은 바꾸지 않고 조건의 적용 대상을 변환된 값으로 바꾼다.',
        'ambiguity_review': '완성 과제나 실행 결과를 대신 요구하지 않는 코드 이해 질문이다. 조건식은 교재 그대로이며 구체 입력의 계산 결과를 새로 만들지 않는다.',
    },
    {
        'pair_id': 'rel-06', 'error_type': '식별 이름과 용도 설명의 필드 연결',
        'messages': [user('기능 카드의 YAML에서 카드 이름과 쓰임새 설명을 다른 칸에 적는 이유가 궁금해요. 폴더 이름과 맞아야 하는 칸을 중심으로 구분해 주세요.')],
        'control_answer': 'name은 카드를 구별하는 이름이고 description은 카드가 무엇을 하며 언제 쓰이는지 설명해요. 폴더 이름과 같아야 하는 것은 name 값이에요.',
        'old': '같아야 하는 것은 name 값', 'new': '같아야 하는 것은 description 값',
        'unsupported': '폴더 이름과 같아야 하는 것은 description 값이에요.',
        'sources': [{'path': 'kb/prompt/PR-02-03.md', 'quote': '기본 정보의 name은 카드를 구별하는 이름이고, description은 카드가 무엇을 하며 언제 쓰이는지 설명해요.'},
                    {'path': 'kb/prompt/PR-02-03.md', 'quote': 'CARD.md의 name은 그 기능 카드 폴더의 이름과 같아야 해요.'}],
        'error_reason': '두 필드의 일반 설명은 그대로 두고 폴더명 일치 의무를 잘못된 필드에 연결한다.',
        'ambiguity_review': '실제 파일이나 임의 카드 이름을 제공하지 않는다. 어떤 description이 우연히 폴더명과 같을 수 있다는 문제가 아니라, 교재가 요구하는 일치 조건을 묻는다.',
    },
    {
        'pair_id': 'rel-07', 'error_type': '비교표의 행과 열 연결',
        'messages': [user('교재의 작업공간 비교표를 읽는 연습을 하고 있어요. 터미널형 행에서 주로 작업하는 모습 열의 칸에는 무엇이라고 되어 있는지 설명해 주세요.')],
        'control_answer': '교재 비교표의 터미널형 행에서 주로 작업하는 모습 칸에는 명령과 실행 결과를 글자로 주고받는다고 되어 있어요.',
        'old': '명령과 실행 결과를 글자로 주고받는다고', 'new': '편집 화면에서 코드를 보며 고친다고',
        'unsupported': '교재 비교표의 터미널형 행에서 주로 작업하는 모습 칸에는 편집 화면에서 코드를 보며 고친다고 되어 있어요.',
        'sources': [{'path': 'kb/prompt/PR-03-01-table1.csv', 'quote': '구분,주로 작업하는 모습,강점'},
                    {'path': 'kb/prompt/PR-03-01-table1.csv', 'quote': '터미널형,명령과 실행 결과를 글자로 주고받아요'},
                    {'path': 'kb/prompt/PR-03-01-table1.csv', 'quote': '편집기형,편집 화면에서 코드를 보며 고쳐요'}],
        'error_reason': '터미널형 행의 주로 작업하는 모습 열에 편집기형 행의 내용을 잘못 연결한다. 질문과 답변 모두 특정 표의 특정 칸을 가리킨다.',
        'ambiguity_review': '실제 도구 기능의 겹침이나 전체 도구에 대한 일반화를 판정하지 않는다. 고정된 교재 표에 무엇이 적혀 있는지만 판정하므로 편집기의 터미널 지원 여부와 무관하다.',
    },
    {
        'pair_id': 'rel-08', 'error_type': '명령 이름과 입력 위치의 관계',
        'messages': [user('교재대로 일반 터미널에서 `helpycode`를 입력해 시작했어요.'),
                     {'role': 'assistant', 'content': '그다음에는 HelpyCode의 입력창에서 대화를 이어가요.'},
                     user('이제 모델을 선택하려고 `/model`을 쓰려는데, 처음 시작 명령을 썼던 곳과 입력 위치가 어떻게 다른지 설명해 주세요.')],
        'control_answer': '`helpycode`는 일반 터미널에서 시작할 때 쓰고, `/model`은 HelpyCode 입력창에 넣어요. 모델 선택 화면이 나오면 이 과정에서 사용하는 모델을 골라요.',
        'old': '`/model`은 HelpyCode 입력창에', 'new': '`/model`은 HelpyCode 입력창이 아닌 일반 터미널에',
        'unsupported': '`/model`은 HelpyCode 입력창이 아닌 일반 터미널에 넣어요.',
        'sources': [{'path': 'kb/prompt/PR-03-02.md', 'quote': '터미널이 `연습장`을 작업 폴더로 사용하도록 맞춘 뒤, `helpycode`를 입력해 HelpyCode를 시작해요.'},
                    {'path': 'kb/prompt/PR-03-02.md', 'quote': 'HelpyCode 입력창에 `/model`을 입력하고 나타나는 선택 화면에서 `Helpy Pro`를 골라요.'}],
        'error_reason': '명령과 모델 선택 기능은 유지하면서 슬래시 명령의 입력 위치만 일반 터미널로 바꾼다.',
        'ambiguity_review': '실제 셸에서 명령을 실행하거나 환경을 가정하지 않는다. 오류 답변은 HelpyCode 입력창이 아니라고 명시적으로 부정하므로 같은 터미널 창 안의 도구 입력창을 일반 터미널이라고 부르는 표현상 혼동을 배제한다.',
    },
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def build(root=ROOT, pairs=PAIRS, expected_kb_sha=KB_SHA, expected_count=8):
    root = Path(root).resolve()
    inventory = {}
    for path in sorted((root / 'kb').rglob('*')):
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(root / 'kb'):
                raise ValueError('Invalid KB path')
            inventory[str(path.relative_to(root))] = digest(path.read_bytes())
    if digest(canonical(inventory)) != expected_kb_sha:
        raise ValueError('KB content or inventory changed')
    if len(pairs) != expected_count or len({pair['pair_id'] for pair in pairs}) != expected_count:
        raise ValueError('Unexpected pair inventory')
    proposals, items = [], []
    for pair in pairs:
        pair = copy.deepcopy(pair)
        pair.setdefault('error_group', 'course_relation')
        if pair['error_group'] not in ('course_relation', 'code_interpretation'):
            raise ValueError('Invalid error group')
        answer, old, new, span = (pair[key] for key in ('control_answer', 'old', 'new', 'unsupported'))
        if (not all(isinstance(v, str) and v for v in (answer, old, new, span))
                or answer.count(old) != 1 or old == new or new in answer):
            raise ValueError('Edit is not a unique semantic substitution')
        wrong = answer.replace(old, new, 1)
        start = answer.index(old)
        if wrong.count(span) != 1:
            raise ValueError('Error span does not uniquely match answer')
        span_start = wrong.index(span)
        if not span_start <= start or span_start + len(span) < start + len(new):
            raise ValueError('Error span does not cover edit')
        if (not pair['messages'] or pair['messages'][-1]['role'] != 'user'
                or any(m.get('role') not in ('user', 'assistant') or not m.get('content') for m in pair['messages'])
                or not pair.get('error_reason') or not pair.get('ambiguity_review') or not pair['sources']):
            raise ValueError('Incomplete manual rationale or conversation')
        for source in pair['sources']:
            path, quote = source['path'], source['quote']
            if path not in inventory or not quote or quote not in (root / path).read_text():
                raise ValueError('Exact support quotation missing from pinned KB')
            source['source_sha256'] = inventory[path]
            source['quote_sha256'] = digest(quote.encode())
            source['line'] = (root / path).read_text().split(quote, 1)[0].count('\n') + 1
        pair.update(counterfactual_answer=wrong, replacement_start=start,
                    unsupported_start=span_start, review_status='pending_parent_review')
        proposals.append(pair)
        for kind, text, grounded, unsupported in (
                ('control', answer, True, []), ('counterfactual', wrong, False, [span])):
            items.append({'id':pair['pair_id'] + '/' + kind, 'endpoint':'guard',
                          'pair_id':pair['pair_id'], 'contrast_kind':kind,
                          'error_group':pair['error_group'],
                          'messages':copy.deepcopy(pair['messages']), 'candidate_answer':text,
                          'labels':{'category':'normal','decision':'allow','grounded':grounded,
                                    'safe':True,'unsupported':unsupported}})
    manifest = {
        'scope':'kb_manual_relational_guard_development', 'pairs':len(pairs),'count':len(items),
        'version':2, 'previous_version':copy.deepcopy(PREVIOUS_VERSION),
        'kb_files':inventory,'kb_manifest_sha256':expected_kb_sha,
        'kb_manifest_hash_formula':'SHA256 UTF-8 JSON object path->file SHA256, ensure_ascii=False, sort_keys=True, separators=(comma,colon)',
        'proposal_sha256':digest(canonical(proposals)),
        'design_constraints':{'source_only':'organizer KB','public_questions_read_for_design':False,
                             'saved_predictions_read_for_design':False,'model_evaluations':0,'api_calls':0,
                             'external_sources_used':False,'fixed_gold_before_any_prediction':True},
        'review_status':'pending_parent_review_before_scoring',
        'training_allowed':False,'deployment_allowed':False,'independent_source_holdout':False,
        'primary_metric':'paired_grounded_accuracy',
        'limitations':['Eight hand-written pairs, not a representative sample or independent holdout.',
                       'Normal/safe input labels are manual proposals pending parent review.',
                       'Wrong spans are subjective; grounded control/detection/pair accuracy is primary.',
                       'Some wrong answers retain true statements and contain an internal contradiction.',
                       'New wording is authored from KB; lexical non-overlap with every public question was not checked.'],
    }
    return proposals, items, manifest


def write(root=ROOT):
    root = Path(root).resolve()
    for name, expected in PREVIOUS_VERSION['files'].items():
        previous = root / PREVIOUS_VERSION['path'] / name
        if digest(previous.read_bytes()) != expected:
            raise ValueError('Preserved previous version changed')
    proposals, items, manifest = build(root)
    data = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items).encode()
    manifest['dataset_sha256'] = digest(data)
    dest = root / DESTINATION
    if not dest.resolve().is_relative_to(root / 'experiments/.cache'):
        raise ValueError('Output must remain in ignored development cache')
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'inputs.jsonl').write_bytes(data)
    (dest / 'proposal.json').write_text(json.dumps(proposals, ensure_ascii=False, indent=2) + '\n')
    (dest / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return dest, manifest


if __name__ == '__main__':
    dest, manifest = write()
    print(json.dumps({'path':str(dest), 'count':manifest['count'], 'dataset_sha256':manifest['dataset_sha256']}))
