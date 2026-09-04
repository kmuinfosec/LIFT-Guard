"""Qwen3Guard-Gen-4B. 생성형 content-safety guard.

user 메시지에 chat template 을 적용하면 안전 라벨을 생성한다. 출력에 'Safe / Unsafe /
Controversial' 형태의 라벨이 들어간다. Unsafe 와 Controversial 을 유해로 본다.

ponytail: 출력 포맷은 모델 카드 기준이다. 서버 첫 실행에서 scripts/00_smoke.py 로 raw
생성을 확인하고 parse_generation 을 맞출 것.
"""

from __future__ import annotations

import re

from .base import LMGuard


class Qwen3Guard(LMGuard):
    def __init__(self, model_id: str = "Qwen/Qwen3Guard-Gen-4B", **kw):
        kw.setdefault("max_new_tokens", 16)
        super().__init__(model_id=model_id, **kw)

    def build_prompt(self, user_text: str) -> str:
        return self._tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True,
        )

    def parse_generation(self, text: str) -> float:
        low = text.lower()
        # 생성 라벨 우선순위: 명시적 Safe 가 있으면 무해, 아니면 unsafe/controversial 탐색
        if re.search(r"\bunsafe\b", low) or re.search(r"\bcontroversial\b", low):
            return 1.0
        if re.search(r"\bsafe\b", low):
            return 0.0
        return 0.0

    def score(self, prompts: list[str]):
        return self.score_by_generation(prompts)
