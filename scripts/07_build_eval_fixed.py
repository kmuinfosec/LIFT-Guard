"""평가 데이터셋 재구성 (수정판).

이전 06_build_eval.py 의 두 결함을 고친다.

  결함 1: negative 의 chunks 를 [] 로 두어 도메인 제거를 건너뛰었다. 04_defense.py 가
          빈 chunks 를 원본으로 대체하므로, positive 만 처리하고 negative 는 원본
          그대로 비교하는 꼴이 됐다.
  결함 2: 토큰 제거 기준값을 이전 negative(benign_substituted)에서 구한 값(0.1836)을
          그대로 썼다. 지금 데이터와 무관하고 test 를 본 값이다.

수정
  - positive/negative 모두 같은 절차로 도메인 제거를 거친다.
  - 기준값은 이 데이터의 train 행에서만 구한다.
  - baseline negative 는 대응하는 시스템 프롬프트가 없어 제거 대상이 아니다.
    원본을 그대로 통과시키고 결과에서 도메인 5종과 분리해 볼 수 있게 표시한다.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from importlib import import_module  # noqa: E402

from sps.data import load_system_prompts, read_dataset  # noqa: E402
from sps.embedders import Embedder  # noqa: E402
from sps.paths import CONTEXT_SLUGS, RAW, RESULTS  # noqa: E402
from sps.sanitizer import SanitizerConfig, SystemPromptSubtractor  # noqa: E402

sweep = import_module("16_subset_sweep")

WINDOW, STRIDE, REF, GRAN, QUANTILE = 8, 3, "summary_v2", "whole", 0.80


def load_negatives() -> list[dict]:
    """정상 질의. 도메인 5종 + baseline."""
    out = []
    for p in sorted((RAW / "benign_normal").glob("*.csv")):
        slug = p.stem.replace("benign_", "")
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            i = int(r["Index"])
            out.append({
                "uid": f"neg-{slug}-{i}", "label": 0, "index": i,
                "category": slug, "context": slug, "strategy": "none",
                "source": "generated-benign",
                # JBB goal 과 무관하므로 홀짝으로 나눈다.
                "is_train": i % 2 == 0,
                "original": r["Goal"].strip(), "goal": r["Goal"].strip(),
                # baseline 은 시스템 프롬프트가 없어 도메인 제거 대상이 아니다.
                "sanitizable": slug in CONTEXT_SLUGS,
            })
    return out


def main() -> None:
    ds = read_dataset()
    train_goals = sweep.category_train([s for s in ds if s.label == 1])
    pos = [{
        "uid": s.uid, "label": 1, "index": s.index, "category": s.category,
        "context": s.context, "strategy": s.strategy, "source": s.source,
        "is_train": s.index in train_goals, "original": s.text, "goal": s.goal,
        "sanitizable": True,
    } for s in ds
        if s.label == 1 and s.context != "none"
        and "Llama-2" in s.target_model and "Qwen3.8" in s.attacker_model]

    rows = pos + load_negatives()
    n_pos = len(pos)
    n_san = sum(r["sanitizable"] for r in rows)
    print(f"positive {n_pos:,} / negative {len(rows) - n_pos:,}")
    print(f"  train {sum(r['is_train'] for r in rows):,} / "
          f"test {sum(not r['is_train'] for r in rows):,}")
    print(f"  도메인 제거 대상 {n_san:,} / 통과(baseline) {len(rows) - n_san:,}")

    emb = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    cfg = SanitizerConfig(window_tokens=WINDOW, stride_tokens=STRIDE, smooth_tokens=5,
                          reference_granularity=GRAN)
    san = SystemPromptSubtractor(emb, load_system_prompts(kind=REF), cfg)

    # 제거 대상만 인코딩한다. baseline 은 참조가 없어 계산할 것이 없다.
    tgt = [r for r in rows if r["sanitizable"]]
    sets = san.encode_windows([r["original"] for r in tgt])

    # 기준값은 train 행에서만 구한다. positive 와 negative 를 함께 쓴다.
    tr = [r["is_train"] for r in tgt]
    thr = san.calibrate_threshold([w for w, m in zip(sets, tr) if m],
                                  [r["context"] for r, m in zip(tgt, tr) if m],
                                  quantile=QUANTILE)
    print(f"  기준값 {thr:.4f} (train {sum(tr):,}행에서 보정)")

    by_uid = {}
    for r, ws in zip(tgt, sets):
        res = san.sanitize(ws, r["context"], thr)
        by_uid[r["uid"]] = (res.kept_chunks, res.keep_ratio)

    out = RESULTS / "eval_set_fixed.jsonl"
    n_chunks = 0
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            chunks, kr = by_uid.get(r["uid"], ([r["original"]], 1.0))
            n_chunks += len(chunks)
            fh.write(json.dumps({**r, "chunks": chunks, "keep_ratio": kr},
                                ensure_ascii=False) + "\n")

    print(f"\n{len(rows):,}행 -> {out}")
    print(f"  채점 단위: original {len(rows):,}, goal {len(rows):,}, chunks {n_chunks:,}")

    import statistics as st
    print("\n잔존율 (도메인 제거 대상만)")
    for lbl, f in (("공격", lambda r: r["label"] == 1),
                   ("정상 도메인 질의", lambda r: r["label"] == 0 and r["sanitizable"])):
        v = [by_uid[r["uid"]][1] for r in rows if f(r) and r["uid"] in by_uid]
        if v:
            print(f"  {lbl:16s} n={len(v):5d}  평균 {st.mean(v):.3f}  중앙 {st.median(v):.3f}")


if __name__ == "__main__":
    main()
