"""실험 A 데이터셋: JBB harmful goals 100 (positive) vs benign goals 100 (negative).

negative 는 그냥 benign 이 아니라 harmful 과 주제를 맞춘 hard negative 다. 같은 index 의
harmful/benign 이 한 단어 차이로 짝지어져 있어(phishing/prompts, ransomware/encrypts),
탐지기가 주제가 아니라 의도로 판별하는지를 시험한다.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA  # noqa: E402

BASE = "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data"


def fetch(name: str) -> list[dict]:
    cache = DATA / f"jbb_{name}.csv"
    if not cache.is_file():
        with urllib.request.urlopen(f"{BASE}/{name}-behaviors.csv", timeout=60) as r:
            cache.write_bytes(r.read())
    return list(csv.DictReader(io.StringIO(cache.read_text(encoding="utf-8-sig"))))


def main() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass

    harmful = fetch("harmful")
    benign = fetch("benign")
    bi = {int(r["Index"]): r for r in benign}

    rows = []
    for h in harmful:
        i = int(h["Index"])
        rows.append({"index": i, "label": 1, "category": h["Category"],
                     "goal": h["Goal"], "source": "jbb_harmful"})
        b = bi.get(i)
        if b:
            rows.append({"index": i, "label": 0, "category": b.get("Category", ""),
                         "goal": b["Goal"], "source": "jbb_benign_hardneg"})

    out = DATA / "content_safety.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "label", "category", "goal", "source"])
        w.writeheader()
        w.writerows(rows)

    n_pos = sum(1 for r in rows if r["label"] == 1)
    print(f"harmful {n_pos} / benign(hard neg) {len(rows) - n_pos} -> {out}")


if __name__ == "__main__":
    main()
