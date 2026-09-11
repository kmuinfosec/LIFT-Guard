"""임계값 보정과 지표.

가드는 연속 점수를 내놓고, 이를 판정으로 바꾸려면 임계값 하나가 필요하다. 가드마다 점수
척도가 크게 다르다 — 우리 실험에서 F1 최적 컷이 어떤 모델은 0.0015, 어떤 모델은 0.77
이었다. 그래서 0.5 같은 공통 기본값을 쓰면 그 척도에 맞지 않는 모델이 조용히 망가진다.
모델마다, 트래픽 성격이 갈리면 성격마다 따로 보정한다.

보정에는 라벨이 붙은 별도 분할을 쓴다. 학습하는 것은 없다.
"""

from __future__ import annotations

import numpy as np


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """precision, recall, F1. 분모가 0 이면 0 으로 둔다."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def metrics(score: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """임계값을 고정했을 때의 precision / recall / f1. 유해가 양성이다."""
    b, y = np.asarray(score) >= threshold, np.asarray(y).astype(bool)
    tp, fp, fn = int((b & y).sum()), int((b & ~y).sum()), int((~b & y).sum())
    p, r, f1 = prf(tp, fp, fn)
    return {"threshold": float(threshold), "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": int((~b & ~y).sum())}


def best_threshold(score: np.ndarray, y: np.ndarray,
                   tie: str = "high") -> tuple[float, float]:
    """F1 이 최대가 되는 임계값과 그때의 F1.

    후보는 분위수 격자가 아니라 **모든 고유 점수**다. 정렬 한 번과 누적합이면 끝나서
    싸고, 최적값을 놓칠 일이 없다.

    `tie="high"` 는 동점일 때 큰 임계값을 고른다 — 같은 F1 이면 양성 판정이 적어 오탐이
    줄기 때문이다. recall 을 우선하려면 `tie="low"` 를 쓴다.
    """
    score, y = np.asarray(score, dtype=np.float64), np.asarray(y).astype(bool)
    order = np.argsort(-score, kind="mergesort")          # 내림차순
    s, yy = score[order], y[order]
    tp = np.cumsum(yy)                                    # 앞에서 k개를 양성으로 볼 때
    fp = np.cumsum(~yy)
    n_pos = int(y.sum())
    # 같은 점수는 한 덩어리로만 자를 수 있다. 각 덩어리의 마지막 위치만 후보로 남긴다.
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp, cut = tp[last], fp[last], s[last]
    f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + (n_pos - tp)), 0.0)
    best = int(np.argmax(f1)) if tie == "high" else int(len(f1) - 1 - np.argmax(f1[::-1]))
    return float(cut[best]), float(f1[best])


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """순위 기반 AUROC. 동점은 평균 순위로 처리한다.

    임계값과 무관한 점검용이다. 0.5 근처면 가드의 출력을 의도대로 파싱하지 못하고 있다는
    뜻이다.
    """
    score, y = np.asarray(score), np.asarray(y).astype(bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
