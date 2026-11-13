import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
from openai import AsyncOpenAI


@dataclass(frozen=True)
class Settings:
    team_token: str
    openai_api_key: str
    openai_base_url: Optional[str]
    openai_model: str
    embedding_api_key: str
    embedding_base_url: Optional[str]
    embedding_model: str
    kb_dir: Path
    public_dir: Path
    assets_dir: Path
    max_concurrency: int = 64
    request_deadline_sec: float = 27.5
    llm_timeout_sec: float = 22.0


def load_settings() -> Settings:
    root = Path(__file__).resolve().parents[1]
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    embed_key = os.environ.get("EMBEDDING_API_KEY", "") or openai_key
    return Settings(
        team_token=os.environ.get("TEAM_TOKEN", "dev-token"),
        openai_api_key=openai_key,
        openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-5-nano"),
        embedding_api_key=embed_key,
        embedding_base_url=os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        kb_dir=Path(os.environ.get("KB_DIR", str(root / "kb"))),
        public_dir=Path(os.environ.get("PUBLIC_DIR", str(root / "public"))),
        assets_dir=Path(os.environ.get("ASSETS_DIR", str(root / "app" / "assets"))),
    )


class ClientPool:
    """Shared AsyncOpenAI clients with 64-concurrency-safe httpx connection pooling."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=64),
            timeout=httpx.Timeout(settings.llm_timeout_sec, connect=4.0, read=settings.llm_timeout_sec, write=5.0, pool=4.0),
        )
        self.chat_client: Optional[AsyncOpenAI] = None
        self.embed_client: Optional[AsyncOpenAI] = None

        if settings.openai_api_key:
            self.chat_client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                http_client=self._http_client,
                max_retries=1,
            )
        if settings.embedding_api_key:
            self.embed_client = AsyncOpenAI(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
                http_client=self._http_client,
                max_retries=1,
            )

    async def close(self) -> None:
        await self._http_client.aclose()
