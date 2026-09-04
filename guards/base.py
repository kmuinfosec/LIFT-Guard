"""Guard 공통 유틸.

대부분의 guard 는 생성형 LLM 이라 "마지막 토큰의 확률"이나 "짧은 생성 텍스트"로 판정한다.
두 방식을 여기에 모아 각 guard 가 프롬프트 구성과 파싱만 정의하면 되게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LMGuard:
    """생성형 LLM 기반 guard 의 공통 뼈대."""

    model_id: str
    device: str | None = None
    dtype: str = "float16"
    batch_size: int = 16
    max_new_tokens: int = 8
    _tok: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"
        self._model = self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM

        return (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=getattr(torch, self.dtype) if self.device == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

    # --- 하위 클래스가 정의 ---
    def build_prompt(self, user_text: str) -> str:
        """user 요청을 guard 의 판정 프롬프트로 감싼다."""
        raise NotImplementedError

    def parse_generation(self, text: str) -> float:
        """생성 텍스트에서 [0,1] 유해 확률을 뽑는다."""
        raise NotImplementedError

    # --- 두 가지 채점 방식 ---
    def score_by_generation(self, prompts: list[str]) -> np.ndarray:
        """짧게 생성한 뒤 텍스트를 파싱한다."""
        import torch

        out = np.zeros(len(prompts), dtype=np.float32)
        # 길이가 제각각이라 그냥 배칭하면 패딩이 배치를 지배한다. 길이순으로 묶고
        # 결과만 원래 자리로 되돌린다.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for i in range(0, len(order), self.batch_size):
            idx = order[i : i + self.batch_size]
            chunk = [self.build_prompt(prompts[j]) for j in idx]
            enc = self._tok(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=4096,
                            add_special_tokens=False).to(self.device)
            with torch.no_grad():
                gen = self._model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                           do_sample=False,
                                           pad_token_id=self._tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            texts = self._tok.batch_decode(new, skip_special_tokens=True)
            for j, t in zip(idx, texts):
                out[j] = self.parse_generation(t)
        return out

    def score_by_token(self, prompts: list[str], pos_word: str, neg_word: str) -> np.ndarray:
        """다음 토큰이 pos_word 일 확률을 neg_word 대비로 정규화한다.

        Granite Guardian 처럼 판정을 단일 토큰(yes/no)으로 내는 모델에 쓴다. 생성보다
        빠르고 연속 점수라 ROC/AUROC 를 그릴 수 있다.
        """
        import torch

        pos = self._tok.encode(pos_word, add_special_tokens=False)[0]
        neg = self._tok.encode(neg_word, add_special_tokens=False)[0]
        out = np.zeros(len(prompts), dtype=np.float32)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for i in range(0, len(order), self.batch_size):
            idx = order[i : i + self.batch_size]
            chunk = [self.build_prompt(prompts[j]) for j in idx]
            enc = self._tok(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=4096,
                            add_special_tokens=False).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits[:, -1, :].float()
            pair = torch.stack([logits[:, neg], logits[:, pos]], dim=-1)
            out[idx] = pair.softmax(-1)[:, 1].cpu().numpy()
        return out

    @property
    def key(self) -> str:
        return type(self).__name__
