"""Offline asset builder for KB + public query embeddings and optional synthetic data.

Usage:
    python scripts/build_offline_assets.py
    python scripts/build_offline_assets.py --with-synthetic
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import ClientPool, Settings, load_settings
from app.normalize import normalize_text


def _load_kb_chunks(settings: Settings) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load KB child chunks and public tutor/guard examples using app.kb_index."""
    from app.kb_index import KBIndex

    idx = KBIndex(settings, embed_client=None)
    chunks = [
        {
            "chunk_id": c.chunk_id,
            "section_id": c.section_id,
            "text": c.text,
        }
        for c in idx.chunks
    ]
    return chunks, idx.tutor_examples, idx.guard_examples


def _extract_user_query(item: Dict[str, Any]) -> str:
    user_turns = [
        normalize_text(m.get("content", ""))
        for m in item.get("messages", [])
        if m.get("role") == "user"
    ]
    return " ".join(t for t in user_turns if t) or " "


async def _embed_texts(
    pool: ClientPool,
    texts: List[str],
    model: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute embeddings in batches using ClientPool.embed_client."""
    assert pool.embed_client is not None
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    async def _one(batch: List[str]) -> List[List[float]]:
        async with pool.semaphore:
            resp = await pool.embed_client.embeddings.create(model=model, input=batch)
            return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]

    results = await asyncio.gather(*(_one(b) for b in batches))
    flat = [vec for batch_vecs in results for vec in batch_vecs]
    arr = np.asarray(flat, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


async def _generate_synthetic_headers(
    pool: ClientPool,
    chunks: List[Dict[str, str]],
    model: str,
    limit: int = 48,
) -> List[Dict[str, str]]:
    """Optionally generate concise contextual search headers via gpt-5-nano."""
    assert pool.chat_client is not None
    # Pick one representative chunk per section up to `limit`
    seen_sections = set()
    targets: List[Dict[str, str]] = []
    for c in chunks:
        sid = c["section_id"]
        if sid not in seen_sections:
            seen_sections.add(sid)
            targets.append(c)
        if len(targets) >= limit:
            break

    async def _one(chunk: Dict[str, str]) -> Dict[str, str]:
        async with pool.semaphore:
            resp = await pool.chat_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "교육 자료 섹션의 검색용 한 줄 요약 헤더(핵심 과목명, 실습/상품/모듈 고유명사 포함, 80자 이내)만 출력하세요."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"[{chunk['section_id']}]\n{chunk['text'][:1200]}",
                    },
                ],
                max_completion_tokens=128,
                extra_body={"reasoning_effort": "low"},
            )
            header = (resp.choices[0].message.content or "").strip()
            return {
                "chunk_id": chunk["chunk_id"],
                "section_id": chunk["section_id"],
                "contextual_header": header,
            }

    return await asyncio.gather(*(_one(t) for t in targets))


async def build_assets(
    settings: Settings,
    with_synthetic: bool = False,
    batch_size: int = 64,
    output_path: Path | None = None,
) -> Path:
    pool = ClientPool(settings)
    try:
        if pool.embed_client is None:
            raise RuntimeError(
                "EMBEDDING_API_KEY (or OPENAI_API_KEY) is not set. "
                "Set the environment variable before running build_offline_assets.py."
            )

        chunks, tutor_examples, guard_examples = _load_kb_chunks(settings)
        if not chunks:
            raise RuntimeError(f"No KB chunks found under {settings.kb_dir}")

        tutor_texts = [_extract_user_query(ex) for ex in tutor_examples]
        guard_texts = [_extract_user_query(ex) for ex in guard_examples]
        print(
            f"Loaded {len(chunks)} KB chunks, {len(tutor_texts)} tutor queries, "
            f"and {len(guard_texts)} guard queries."
        )

        chunk_texts = [c["text"] for c in chunks]
        chunk_emb = await _embed_texts(pool, chunk_texts, settings.embedding_model, batch_size=batch_size)
        tutor_emb = (
            await _embed_texts(pool, tutor_texts, settings.embedding_model, batch_size=batch_size)
            if tutor_texts
            else np.zeros((0, chunk_emb.shape[1]), dtype=np.float32)
        )
        guard_emb = (
            await _embed_texts(pool, guard_texts, settings.embedding_model, batch_size=batch_size)
            if guard_texts
            else np.zeros((0, chunk_emb.shape[1]), dtype=np.float32)
        )

        npz_payload: Dict[str, Any] = {
            "chunk_ids": np.array([c["chunk_id"] for c in chunks]),
            "chunk_section_ids": np.array([c["section_id"] for c in chunks]),
            "chunk_embeddings": chunk_emb,
            "tutor_ids": np.array([str(ex.get("id", "")) for ex in tutor_examples]),
            "tutor_embeddings": tutor_emb,
            "guard_ids": np.array([str(ex.get("id", "")) for ex in guard_examples]),
            "guard_embeddings": guard_emb,
        }

        settings.assets_dir.mkdir(parents=True, exist_ok=True)
        if with_synthetic:
            if pool.chat_client is None:
                raise RuntimeError("OPENAI_API_KEY is required when --with-synthetic is set.")
            headers = await _generate_synthetic_headers(pool, chunks, settings.openai_model)
            npz_payload["contextual_chunk_ids"] = np.array([h["chunk_id"] for h in headers])
            npz_payload["contextual_headers"] = np.array([h["contextual_header"] for h in headers])
            syn_file = settings.assets_dir / "synthetic_headers.json"
            syn_file.write_text(json.dumps(headers, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved {len(headers)} synthetic contextual headers to {syn_file}")

        out_file = output_path or (settings.assets_dir / "kb_embeddings.npz")
        np.savez_compressed(out_file, **npz_payload)
        print(
            f"Saved embeddings to {out_file} "
            f"(chunks={chunk_emb.shape}, tutor={tutor_emb.shape}, guard={guard_emb.shape})"
        )

        return out_file
    finally:
        await pool.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build offline KB & public query embedding assets.")
    parser.add_argument("--with-synthetic", action="store_true", help="Generate contextual headers via gpt-5-nano.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--output", type=Path, default=None, help="Output .npz path.")
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        asyncio.run(
            build_assets(
                settings,
                with_synthetic=args.with_synthetic,
                batch_size=args.batch_size,
                output_path=args.output,
            )
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
