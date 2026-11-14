import json
import math
import os
from pathlib import Path

from openai import AsyncOpenAI
from knowledge_base import load_knowledge_base


class KnowledgeSearch:
    def __init__(self):
        path = Path(__file__).resolve().parent / "work/kb_index.json"
        self.index = json.loads(path.read_text(encoding="utf-8"))
        self.sections = load_knowledge_base()

        self.model = os.environ["EMBEDDING_MODEL"]
        base_url = os.environ["EMBEDDING_BASE_URL"]

        if self.index["model"] != self.model:
            raise ValueError("모델이 변경됐습니다. 색인을 다시 생성하세요.")
        if self.index["base_url"].rstrip("/") != base_url.rstrip("/"):
            raise ValueError("API 주소가 변경됐습니다. 색인을 다시 생성하세요.")

        # 저장된 조각이 현재 자료에도 존재하는지 기본 검사합니다.
        for chunk in self.index["chunks"]:
            section = self.sections.get(chunk["section_id"])
            if section is None or chunk["text"] not in section["text"]:
                raise ValueError("자료가 변경됐습니다. 색인을 다시 생성하세요.")

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=os.environ["EMBEDDING_API_KEY"],
            timeout=15,
            max_retries=0,
        )

    async def search(self, query, top_k=5):
        if not query.strip() or top_k <= 0:
            return []

        response = await self.client.embeddings.create(
            model=self.model,
            input=[query],
            encoding_format="float",
        )
        vector = response.data[0].embedding

        if len(vector) != self.index["dimension"]:
            raise ValueError("질문과 자료의 벡터 차원이 다릅니다.")
        if not all(math.isfinite(v) for v in vector):
            raise ValueError("질문 벡터가 유효하지 않습니다.")

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            raise ValueError("질문 벡터가 영벡터입니다.")
        vector = [v / norm for v in vector]

        # 같은 섹션의 조각은 가장 높은 점수 하나로 합칩니다.
        scores = {}
        for chunk in self.index["chunks"]:
            score = sum(a * b for a, b in zip(vector, chunk["vector"]))
            sid = chunk["section_id"]
            scores[sid] = max(scores.get(sid, -float("inf")), score)

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

        # 검증 모델에는 조각뿐 아니라 해당 섹션 전체를 제공합니다.
        return [{**self.sections[sid], "score": score} for sid, score in ranked]

    async def close(self):
        await self.client.close()
