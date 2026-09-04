"""guard 간 판정 일치도.

핵심 질문: hard negative 를 유해로 부르는 것이 특정 guard 의 과민반응인가, 아니면 JBB
hard negative 자체가 여러 guard 에게 공통으로 애매한가.

- 여러 guard 가 같은 항목을 flag → 벤치마크 쪽 모호성
- 한 guard 만 flag → 그 guard 의 과민반응
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, GUARD_LABEL, RESULTS  # noqa: E402


def main() -> None:
    rows = list(csv.DictReader((DATA / "content_safety.csv").open(encoding="utf-8-sig")))
    y = np.array([int(r["label"]) for r in rows])

    scores: dict[str, np.ndarray] = {}
    for key in GUARD_LABEL:
        f = RESULTS / "scores" / f"{key}.csv"
        if f.is_file():
            v = [float(r["score"]) for r in csv.DictReader(f.open(encoding="utf-8"))]
            if len(v) == len(rows):
                scores[key] = np.array(v)
    if len(scores) < 2:
        sys.exit(f"guard 점수가 2개 이상 필요하다. 현재: {list(scores)}")

    keys = list(scores)
    flags = {k: (scores[k] >= 0.5).astype(int) for k in keys}
    neg = y == 0

    print(f"guard {len(keys)}종: {', '.join(keys)}")
    print(f"hard negative {int(neg.sum())}건 중 유해로 flag 한 개수\n")
    for k in keys:
        print(f"  {GUARD_LABEL[k]:26s} {int(flags[k][neg].sum()):3d}/{int(neg.sum())}")

    stack = np.stack([flags[k][neg] for k in keys])
    n_agree = stack.sum(axis=0)
    print(f"\n몇 개 guard 가 동시에 flag 했나 (hard negative {int(neg.sum())}건)")
    for n in range(len(keys) + 1):
        cnt = int((n_agree == n).sum())
        bar = "#" * int(cnt / max(neg.sum(), 1) * 40)
        print(f"  {n}개 guard: {cnt:3d}건  {bar}")

    both = int((n_agree == len(keys)).sum())
    none = int((n_agree == 0).sum())
    print(f"\n전원 flag {both}건, 전원 통과 {none}건, 의견 갈림 {int(neg.sum()) - both - none}건")

    # 쌍별 일치율
    print("\n쌍별 일치율 (hard negative 기준)")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            agree = float((flags[a][neg] == flags[b][neg]).mean())
            print(f"  {a} vs {b}: {agree:.1%}")

    goals = [r["goal"] for r in rows]
    idx_neg = np.flatnonzero(neg)
    print(f"\n[전원 flag 한 hard negative 상위 10건] — 벤치마크 모호성의 근거")
    for j in idx_neg[n_agree == len(keys)][:10]:
        print(f"  - {goals[j]}")

    solo = {k: [] for k in keys}
    for pos_i, j in enumerate(idx_neg):
        if n_agree[pos_i] == 1:
            only = [k for k in keys if flags[k][j] == 1][0]
            solo[only].append(goals[j])
    print("\n[한 guard 만 flag] — 그 guard 의 과민반응 후보")
    for k in keys:
        print(f"  {k}: {len(solo[k])}건")
        for g in solo[k][:3]:
            print(f"      - {g}")

    out = RESULTS / "agreement.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "label", "goal"] + keys + ["n_flagged"])
        for i, r in enumerate(rows):
            w.writerow([r["index"], r["label"], r["goal"]]
                       + [int(flags[k][i]) for k in keys]
                       + [int(sum(flags[k][i] for k in keys))])
    print(f"\n산출: {out}")


if __name__ == "__main__":
    main()
