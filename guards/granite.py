"""IBM Granite Guardian 3.0 2B. 판정을 yes/no 단일 토큰으로 낸다.

로컬 검증 완료된 형식이다(2026 CISC-W 실험에서 사용).
"""

from __future__ import annotations

from .base import LMGuard


class GraniteGuard(LMGuard):
    model_id: str = "ibm-granite/granite-guardian-3.0-2b"
    risk: str = "harm"

    def __init__(self, model_id: str = "ibm-granite/granite-guardian-3.0-2b", **kw):
        super().__init__(model_id=model_id, **kw)

    def build_prompt(self, user_text: str) -> str:
        return self._tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            guardian_config={"risk_name": self.risk},
            tokenize=False, add_generation_prompt=True,
        )

    def parse_generation(self, text: str) -> float:  # 미사용, 토큰 채점을 쓴다
        return 1.0 if "yes" in text.lower() else 0.0

    def score(self, prompts: list[str]):
        return self.score_by_token(prompts, "Yes", "No")
