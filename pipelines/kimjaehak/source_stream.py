"""Source-only prefix acceptance within the existing selector deadline.

The first allowed section wins, even if unseen trailing JSON duplicates it.
No response/request text is retained outside this call or written to logs.
"""
import asyncio
import json
from contextlib import asynccontextmanager

from pipelines.kimjaehak.focused_excerpt import Selected as SelectedFocused


_DECODER = json.JSONDecoder()
_JSON_WS = ' \t\r\n'


def first_section(text, ids):
    """Return an allowed first top-level string field only after its delimiter.

None means keep consuming, not choose an alternative ID. Complete invalid
prefixes intentionally use the same full-response fallback as incomplete ones.
"""
    pos = len(text) - len(text.lstrip(_JSON_WS))
    if pos >= len(text) or text[pos] != '{':
        return None
    pos += 1
    while pos < len(text) and text[pos] in _JSON_WS:
        pos += 1
    try:
        key, pos = _DECODER.raw_decode(text, pos)
        if key != 'section':
            return None
        while pos < len(text) and text[pos] in _JSON_WS:
            pos += 1
        if pos >= len(text) or text[pos] != ':':
            return None
        pos += 1
        while pos < len(text) and text[pos] in _JSON_WS:
            pos += 1
        value, pos = _DECODER.raw_decode(text, pos)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, str) or value not in ids:
        return None
    while pos < len(text) and text[pos] in _JSON_WS:
        pos += 1
    return value if pos < len(text) and text[pos] in ',}' else None


@asynccontextmanager
async def closing_stream(stream):
    """Keep caller cancellation authoritative if transport close also fails."""
    cancelled = None
    try:
        async with stream:
            try:
                yield stream
            except asyncio.CancelledError as exc:
                cancelled = exc
                raise
    except BaseException:
        if cancelled is not None:
            raise cancelled
        raise


class Pipeline(SelectedFocused):
    source_stream_early_finish = True
    source_stream_buffer_chars = 16000

    async def _source_response(self, state, ids, timeout):
        # The parent already passes min(8, remaining); also clamp direct use.
        limit = min(timeout, self.source_extract_timeout, 8)
        if limit <= 0:
            raise TimeoutError('Source budget exhausted')

        async def consume():
            stream = await self.llm._chat.chat.completions.create(
                model=self.llm.model,
                messages=[{'role': 'system', 'content': self.source_extract_system},
                          {'role': 'user', 'content': json.dumps(
                              self._source_payload(state, ids), ensure_ascii=False)}],
                extra_body={'reasoning_effort': 'low'},
                max_completion_tokens=2000, stream=True,
                response_format={'type': 'json_object'})
            text = ''
            async with closing_stream(stream):
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    if len(chunk.choices) != 1:
                        raise ValueError('Unexpected source choices')
                    choice = chunk.choices[0]
                    if getattr(choice, 'index', 0) != 0:
                        raise ValueError('Unexpected source choice index')
                    delta = choice.delta
                    if (getattr(delta, 'refusal', None)
                            or getattr(delta, 'tool_calls', None)
                            or getattr(delta, 'function_call', None)
                            or getattr(choice, 'finish_reason', None)
                            in {'content_filter', 'tool_calls', 'function_call'}):
                        raise ValueError('Non-text source response')
                    content = getattr(delta, 'content', None)
                    if content is None:
                        continue
                    if not isinstance(content, str):
                        raise ValueError('Non-text source content')
                    # Bound each accumulated response before parsing. Oversize
                    # deltas fail closed, including any valid-looking prefix.
                    if len(text) + len(content) > self.source_stream_buffer_chars:
                        raise ValueError('Source response exceeds buffer limit')
                    text += content
                    selected = first_section(text, ids)
                    if selected is not None:
                        # This is derived selection JSON, not the raw completed
                        # model response. Later duplicate keys are never read.
                        return json.dumps({'section': selected}, ensure_ascii=False,
                                          separators=(',', ':'))
            return text.strip()

        # Covers create, reading, parsing and normal context-manager cleanup.
        # As with the existing helper, slow cancellation cleanup can overshoot;
        # no detached shield/background task or additional call is introduced.
        return await asyncio.wait_for(consume(), timeout=limit)
