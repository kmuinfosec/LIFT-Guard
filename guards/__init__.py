"""Content-safety guard 래퍼.

각 guard 는 `score(prompts) -> np.ndarray` 로 [0,1] 유해 확률을 반환한다. 판정 형식이
모델마다 달라(yes/no 토큰 확률, Safe/Unsafe 생성, 라벨 분류) 그 차이를 여기서 흡수한다.

GPU 의존성(torch, transformers)은 지연 임포트한다. 목록만 볼 때는 임포트가 안 걸린다.
"""

from __future__ import annotations

import os

try:  # 사내 TLS 프록시 대응. 없어도 동작한다.
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

#: 실험 A 에서 비교할 content-safety guard. 사용자가 지정한 우선순위 순서.
REGISTRY = {
    "granite": "guards.granite:GraniteGuard",
    "qwen3guard": "guards.qwen3guard:Qwen3Guard",
    "wildguard": "guards.wildguard:WildGuard",
    "nemotron": "guards.nemotron:NemotronGuard",
}


def load(key: str, **kw):
    """레지스트리 키로 guard 를 생성한다."""
    import importlib

    module_path, cls_name = REGISTRY[key].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**kw)
