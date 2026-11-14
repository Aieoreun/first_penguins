import hashlib
import json
import math
import os
from pathlib import Path

from openai import OpenAI
from knowledge_base import load_knowledge_base


def main():
    kb = load_knowledge_base()
    model = os.environ["EMBEDDING_MODEL"]
    base_url = os.environ["EMBEDDING_BASE_URL"]

    chunks = []
    for section_id, section in sorted(kb.items()):
        text = section["text"]
        start = 0

        while start < len(text):
            end = min(start + 2000, len(text))
            chunks.append(
                {
                    "section_id": section_id,
                    "text": text[start:end],
                }
            )
            if end == len(text):
                break
            start = end - 200  # 조각 경계의 문맥을 일부 겹칩니다.

    signature = hashlib.sha256(
        json.dumps(
            {"model": model, "base_url": base_url, "chunks": chunks},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    target = Path(__file__).resolve().parent / "work/kb_index.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved.get("signature") == signature:
            print("자료·설정 변경 없음: 기존 색인 재사용")
            print("저장 위치:", target)
            return

    client = OpenAI(
        base_url=base_url,
        api_key=os.environ["EMBEDDING_API_KEY"],
        timeout=60,
        max_retries=0,
    )

    records = []
    dimension = None

    for start in range(0, len(chunks), 8):
        batch = chunks[start : start + 8]
        response = client.embeddings.create(
            model=model,
            input=[item["text"] for item in batch],
            encoding_format="float",
        )
        data = sorted(response.data, key=lambda item: item.index)

        if [item.index for item in data] != list(range(len(batch))):
            raise ValueError("임베딩 응답 개수 또는 순서가 잘못됐습니다.")

        for chunk, item in zip(batch, data):
            vector = item.embedding

            if not vector or not all(math.isfinite(v) for v in vector):
                raise ValueError("유효하지 않은 임베딩입니다.")

            dimension = dimension or len(vector)
            if len(vector) != dimension:
                raise ValueError("벡터 차원이 일치하지 않습니다.")

            norm = math.sqrt(sum(v * v for v in vector))
            if norm == 0:
                raise ValueError("영벡터가 반환됐습니다.")

            records.append(
                {
                    **chunk,
                    "vector": [v / norm for v in vector],
                }
            )

        print(f"임베딩 완료: {len(records)}/{len(chunks)}", flush=True)

    result = {
        "signature": signature,
        "model": model,
        "base_url": base_url,
        "dimension": dimension,
        "section_count": len(kb),
        "chunks": records,
    }

    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(target)

    print("색인 생성 완료")
    print("섹션 수:", len(kb))
    print("조각 수:", len(records))
    print("벡터 차원:", dimension)
    print("저장 위치:", target)


if __name__ == "__main__":
    main()
