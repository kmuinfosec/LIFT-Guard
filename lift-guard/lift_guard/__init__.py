"""LIFT-Guard — 도메인 위장 탈옥에서 가드 정확도를 되살리는 슬라이딩 윈도우 채점.
학습하지 않는다.

    from lift_guard import LiftGuard

    guard = LiftGuard(window=64)                 # 기본 가드는 Prompt Guard 2 86M
    guard.calibrate(val_prompts, val_labels)     # 임계값 하나만 정함, 학습 없음
    guard.predict(["...prompt..."])
"""

from .calibrate import auroc, best_threshold, metrics, prf
from .guard import LiftGuard
from .windowing import split_windows

__all__ = ["LiftGuard", "split_windows", "best_threshold", "metrics", "auroc", "prf"]
__version__ = "0.1.0"
