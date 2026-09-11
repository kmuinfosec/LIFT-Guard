"""LIFT-Guard — 공개된 가드 모델 위에 슬라이딩 윈도우 채점을 얹는다.

문제: 도메인 위장 탈옥은 유해 요청을 길고 그럴듯한 업무 프롬프트 안에 숨긴다. 유해한
부분은 전체의 일부일 뿐이라, 프롬프트를 한 덩어리로 읽는 가드는 그 신호를 희석시킨다.

해법: 프롬프트를 겹치는 창으로 잘라 각각 채점하고, 원문 전체 점수와 함께 최댓값을 쓴다.

    score(prompt) = max(guard(prompt), max_i guard(window_i))

원문 항이 중요하다. 창 하나에 담기지 않는 문맥이 있어야 유해하다고 읽히는 프롬프트가
있어서, 그 항을 빼면 그런 건들을 놓친다. 우리 벤치마크에서는 시험한 모든 창 크기에서
결합 점수가 조각만 쓴 점수보다 좋았다.

학습도 파인튜닝도 없다. 가드는 공개된 그대로 쓰고, 정하는 값은 판정 임계값 하나뿐이다
(`calibrate.py` 참고).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .windowing import BOUNDARY_TOKENIZER, split_many

#: 분류기가 라벨 이름을 내놓을 때 '유해' 로 볼 이름들. 소문자 부분일치.
POSITIVE_LABELS = ("unsafe", "harm", "inject", "jail", "malicious")


@dataclass
class LiftGuard:
    """시퀀스 분류 가드를 감싸 슬라이딩 윈도우로 채점한다.

    Args:
        model_id: 가드의 HuggingFace id. `meta-llama/Llama-Prompt-Guard-2-86M` 로
            시험했다.
        window: 창 크기(외부 토크나이저 기준 토큰 수). 우리 실험에서는 64 가 가장
            좋았다. sweep 결과는 README 에 있다.
        stride_ratio: 창 대비 stride 비율. 0.5 면 50% 겹친다.
        include_full: 원문 전체도 함께 채점해 최댓값을 취할지. 켜 두는 편이 좋다.
        batch_size: 채점 배치 크기. 속도에만 영향을 주고, fp16 에서는 점수 끝자리가
            조금 흔들린다.
        max_length: 한 번의 순전파에서 자를 길이. 생략하면 모델 자신의 컨텍스트 길이.
    """

    model_id: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    window: int = 64
    stride_ratio: float = 0.5
    include_full: bool = True
    batch_size: int = 64
    max_length: int | None = None
    device: str | None = None
    boundary_tokenizer: str = BOUNDARY_TOKENIZER
    #: `calibrate` 가 채운다. `predict` 에 필요하다.
    threshold: float | None = None

    _tok: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)
    _positive: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        # 분류 헤드는 hidden_states[:, 0], 즉 [CLS] 자리를 읽는다. 왼쪽 패딩을 쓰면 그
        # 자리가 PAD 토큰이 되고, 점수가 같은 배치에 무엇이 들어 있느냐에 따라 달라진다.
        # 오른쪽 패딩이면 배치 채점이 단건 채점과 같아진다.
        self._tok.padding_side = "right"
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, dtype=torch.float32,
        ).to(self.device).eval()

        id2label = {int(k): str(v) for k, v in self._model.config.id2label.items()}
        self._positive = [i for i, lab in id2label.items()
                          if any(k in lab.lower() for k in POSITIVE_LABELS)]
        if not self._positive:            # LABEL_0/LABEL_1 처럼 이름이 없는 헤드
            self._positive = [max(id2label)]
        if self.max_length is None:
            self.max_length = int(getattr(self._model.config,
                                          "max_position_embeddings", 512))

    # -------------------------------------------------------------------- 채점
    def score_raw(self, texts: list[str]) -> np.ndarray:
        """창 분할 없이, 각 텍스트의 유해 확률."""
        import torch

        out = np.zeros(len(texts), dtype=np.float32)
        # 배칭 전에 길이순으로 정렬한다. 20 토큰짜리와 600 토큰짜리를 한 배치에 섞으면
        # 패딩이 배치를 지배한다.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.no_grad():
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                chunk = [texts[j] if texts[j].strip() else " " for j in idx]
                enc = self._tok(chunk, return_tensors="pt", padding=True,
                                truncation=True,
                                max_length=self.max_length).to(self.device)
                probs = self._model(**enc).logits.softmax(-1)
                out[idx] = probs[:, self._positive].sum(-1).float().cpu().numpy()
        return out

    def score(self, texts: str | list[str]) -> np.ndarray:
        """LIFT-Guard 점수 — 창 조각들과 원문 전체의 최댓값."""
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        stride = max(1, int(round(self.window * self.stride_ratio)))

        groups = split_many(texts, self.window, stride, self.boundary_tokenizer)
        if self.include_full:
            groups = [[t] + g for t, g in zip(texts, groups)]

        flat, bounds = [], []
        for g in groups:
            bounds.append((len(flat), len(flat) + len(g)))
            flat.extend(g)
        raw = self.score_raw(flat)
        out = np.array([raw[a:b].max() for a, b in bounds], dtype=np.float32)
        return out[0] if single else out

    # -------------------------------------------------------------------- 판정
    def predict(self, texts: str | list[str]) -> np.ndarray | bool:
        """유해로 판정되면 True. 임계값이 필요하다."""
        if self.threshold is None:
            raise RuntimeError(
                "임계값이 없다. 라벨이 붙은 프롬프트로 calibrate() 를 부르거나 "
                "guard.threshold 에 직접 넣을 것. 가드마다 점수 척도가 자릿수 단위로 "
                "달라 안전한 기본값이 없다."
            )
        s = self.score(texts)
        return bool(s >= self.threshold) if np.isscalar(s) else s >= self.threshold

    def calibrate(self, texts: list[str], labels, **kw) -> float:
        """주어진 라벨 데이터에서 F1 이 최대가 되는 임계값을 고른다."""
        from .calibrate import best_threshold

        t, _ = best_threshold(self.score(texts), np.asarray(labels), **kw)
        self.threshold = t
        return t
