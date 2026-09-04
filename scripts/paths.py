"""경로와 실험 설정."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

#: 사용자가 지정한 평가 우선순위. granite 는 기준으로 항상 포함한다.
GUARD_ORDER = ["granite", "qwen3guard", "wildguard", "nemotron"]

GUARD_LABEL = {
    "granite": "Granite Guardian 3.0 2B",
    "qwen3guard": "Qwen3Guard-Gen-4B",
    "wildguard": "WildGuard 7B",
    "nemotron": "Nemotron-CS 8B",
}

for _d in (DATA, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
