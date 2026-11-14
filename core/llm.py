"""운영진 제공 API 호출. 채팅(gpt-5-nano)과 임베딩 두 가지뿐이다(SPEC 7.1절).

재시도하지 않는다. 제한 시간을 넘거나 실패하면 예외를 내고, 부른 쪽이 대신할 동작을 정한다.
요청·응답 본문은 로그에 남기지 않는다.
"""
import asyncio
import os
from typing import List, Optional, Sequence

from openai import AsyncOpenAI


class LLM:
    def __init__(self):
        self.model = os.environ["OPENAI_MODEL"]
        self.embedding_model = os.environ.get("EMBEDDING_MODEL", "")
        self._chat = AsyncOpenAI(timeout=25, max_retries=0)  # OPENAI_BASE_URL, OPENAI_API_KEY를 읽는다
        base = os.environ.get("EMBEDDING_BASE_URL", "")
        key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self._embed = AsyncOpenAI(base_url=base, api_key=key, timeout=25, max_retries=0) \
            if base and key and self.embedding_model else None

    @property
    def can_embed(self) -> bool:
        return self._embed is not None

    async def chat(self, system: str, user: str, *, effort: str, max_tokens: int,
                   timeout: float, json_mode: bool = False) -> str:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=max_tokens,
            extra_body={"reasoning_effort": effort},
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = await asyncio.wait_for(self._chat.chat.completions.create(**kwargs), timeout)
        return (r.choices[0].message.content or "").strip()

    async def embed(self, texts: Sequence[str], timeout: float) -> Optional[List[List[float]]]:
        if self._embed is None:
            return None
        r = await asyncio.wait_for(
            self._embed.embeddings.create(model=self.embedding_model, input=list(texts)), timeout)
        return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
