"""평가 데이터셋 재구성.

positive : guard_eval_subset.jsonl 의 공격 프롬프트 3,500건 (도메인 위장 탈옥).
negative : data/benign_queries/*.csv 의 실제 정상 질의 4,200건 (도메인별 700 + baseline 700).

이전 negative 는 공격 프롬프트의 goal 만 무해로 치환한 hard negative 였다. 표면 형태가
positive 와 같아 guard 가 전략 래퍼에 반응하면 그대로 FP 가 됐다. 그 설계는 '의도로만
분류하는가'를 재는 데는 맞지만 실제 서비스 트래픽과는 다르다. 여기서는 negative 를
위장도 전략도 없는 평범한 도메인 질의로 바꾼다.

산출 스키마는 guard_eval_subset.jsonl 과 같아서 05_rewrite / 04_defense 가 그대로 읽는다.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA  # noqa: E402


def main() -> None:
    src = DATA / "guard_eval_subset.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_rows = [r for r in rows if r["label"] == 1]

    for p in sorted((DATA / "benign_queries").glob("*.csv")):
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            i, ctx, goal = int(r["Index"]), r["Behavior"], r["Goal"].strip()
            out_rows.append({
                "uid": f"neg-{ctx}-{i}", "label": 0, "index": i,
                "category": ctx, "context": ctx, "strategy": "none",
                "source": r.get("Source", "generated-benign"),
                # 임계값 고정(0.5)이라 학습은 없다. 홀짝으로 test 절반만 떼어 둔다.
                "is_train": i % 2 == 0,
                "original": goal, "goal": goal, "chunks": [], "keep_ratio": 1.0,
            })

    dst = DATA / "eval_set.jsonl"
    with dst.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in out_rows if r["label"] == 1)
    n_te = sum(1 for r in out_rows if not r["is_train"])
    print(f"{len(out_rows):,}행 (pos {n_pos:,} / neg {len(out_rows) - n_pos:,}), "
          f"test {n_te:,} -> {dst}")
    ctxs = sorted({r["context"] for r in out_rows})
    missing = [c for c in ctxs if not (DATA / "system_prompts" / f"{c}.txt").is_file()]
    print(f"  context {ctxs}")
    if missing:
        sys.exit(f"시스템 프롬프트 없음: {missing}")


if __name__ == "__main__":
    main()
