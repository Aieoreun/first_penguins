"""공개 guard 데이터 전용 단계 추적. 운영 서버에서 import하거나 사용하지 않는다.

실행: python experiments/public_guard_trace.py --output work/public_guard_trace.jsonl
요청·답변 원문이 저장되므로 공개 데이터 이외의 요청에 사용하지 않는다.
"""
import argparse
import asyncio
import contextvars
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.kb import load_chunks
from core.llm import LLM
from core.search import Searcher, load_vectors
from pipelines.kimjaehak.config import configure

# Match server startup: set pinned controls before importing the pipeline.
configure()
from pipelines.kimjaehak.parallel_guard import Pipeline

TRACE = contextvars.ContextVar('public_guard_trace', default=None)
ROLE = contextvars.ContextVar('public_guard_role', default='other')


def event(stage, status, **data):
    trace = TRACE.get()
    if trace is not None:
        trace.append({'stage': stage, 'status': status, 'time_seconds': round(time.monotonic() - trace.start, 4), **data})


class Events(list):
    def __init__(self):
        super().__init__()
        self.start = time.monotonic()


class TracedLLM:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def chat(self, system, user, **kwargs):
        role = ROLE.get()
        started = time.monotonic()
        event('gpt_call', 'start', role=role, system=system, user=user, settings=kwargs)
        try:
            result = await self.inner.chat(system, user, **kwargs)
            event('gpt_call', 'complete', role=role, output=result, elapsed_seconds=round(time.monotonic()-started, 4))
            return result
        except asyncio.CancelledError:
            event('gpt_call', 'cancelled', role=role, elapsed_seconds=round(time.monotonic()-started, 4))
            raise
        except Exception as exc:
            event('gpt_call', 'failed', role=role, error=type(exc).__name__, elapsed_seconds=round(time.monotonic()-started, 4))
            raise


class TracedPipeline(Pipeline):
    async def _search(self, state, extra=''):
        event('search', 'start')
        try:
            await super()._search(state, extra)
            event('search', 'complete', results=[{'section': c.section_id, 'text': c.text, 'score': float(c.score)} for c in state.chunks])
        except BaseException as exc:
            event('search', 'failed', error=type(exc).__name__)
            raise

    async def _ask(self, state, name, system, user, effort, timeout):
        token = ROLE.set(name)
        event(name, 'start', timeout=timeout)
        try:
            result = await super()._ask(state, name, system, user, effort, timeout)
            event(name, 'complete', result=result)
            return result
        except BaseException as exc:
            event(name, 'interrupted', error=type(exc).__name__)
            raise
        finally:
            ROLE.reset(token)

    async def _vote_group(self, state, name, system, check):
        event('vote_'+name, 'start')
        result = await super()._vote_group(state, name, system, check)
        event('vote_'+name, 'complete', votes=result)
        return result

    async def _classify(self, state):
        event('input_classification', 'start')
        await super()._classify(state)
        event('input_classification', 'complete', label=state.label.value if state.label else None,
              decision=state.decision.value, category=state.category.value, debug=state.debug, errors=list(state.errors))

    async def _ground_response(self, payload):
        token = ROLE.set('ground')
        try:
            return await super()._ground_response(payload)
        finally:
            ROLE.reset(token)

    async def _parallel_ground(self, state):
        event('ground', 'start')
        try:
            await super()._parallel_ground(state)
            event('ground', 'complete', grounded=state.grounded, unsupported=state.unsupported, errors=list(state.errors))
        except BaseException as exc:
            event('ground', 'interrupted', error=type(exc).__name__)
            raise

    async def _parallel_safety(self, state):
        token = ROLE.set('safety')
        event('safety', 'start')
        try:
            await super()._parallel_safety(state)
            event('safety', 'complete', safe=state.safe, errors=list(state.errors))
        finally:
            ROLE.reset(token)


def compare(labels, response):
    actual = response.get('input', {})
    output = response.get('output', {})
    return {key: {'expected': labels.get(key), 'actual': (actual if key in ('decision','category') else output).get(key),
                  'match': labels.get(key) == (actual if key in ('decision','category') else output).get(key)}
            for key in ('decision','category','grounded','unsupported','safe') if key in labels}


async def run(args):
    # Fixed input path: cannot be used as a private-request logging endpoint.
    rows = [json.loads(line) for line in (ROOT / 'public' / 'guard.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    llm = LLM()
    chunks = load_chunks(ROOT / 'kb')
    vectors = load_vectors(ROOT / 'index' / 'kb_embeddings.json', llm.embedding_model, chunks)
    pipe = TracedPipeline(Searcher(chunks, vectors), TracedLLM(llm))
    semaphore = asyncio.Semaphore(args.workers)
    output = Path(args.output).resolve()
    allowed = (ROOT / 'work').resolve()
    if not output.is_relative_to(allowed) or output.suffix != '.jsonl':
        raise ValueError('출력은 work/ 아래의 .jsonl 파일이어야 합니다')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('기존 파일이 있습니다. 새 출력 파일을 지정하세요.')
    async def one(row):
        async with semaphore:
            trace = Events()
            token = TRACE.set(trace)
            try:
                body = {'messages': row['messages']}
                if 'candidate_answer' in row:
                    body['candidate_answer'] = row['candidate_answer']
                event('request', 'start', has_candidate='candidate_answer' in body)
                state = await pipe.guard(body)
                response = state.to_guard_response()
                event('final_response', 'complete', response=response, errors=state.errors)
                record = {'id': row['id'], 'messages': row['messages'], 'candidate_answer': row.get('candidate_answer'),
                          'labels': row['labels'], 'events': list(trace), 'response': response,
                          'comparison': compare(row['labels'], response)}
                with output.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                print(f"완료: {row['id']} / 단계 {len(trace)}", flush=True)
            finally:
                TRACE.reset(token)
    await asyncio.gather(*(one(row) for row in rows))
    print(f'저장 완료: {output} / {len(rows)}개')


def main():
    parser = argparse.ArgumentParser(description='공개 guard 데이터만 단계별 추적')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='0이면 공개 데이터 전체')
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error('workers는 1 이상, limit은 0 이상이어야 합니다')
    asyncio.run(run(args))

if __name__ == '__main__':
    main()
