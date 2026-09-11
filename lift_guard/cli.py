"""명령줄 인터페이스 — 두 가지 모드.

    predict    라벨 없이 판정만 한다. 프롬프트마다 점수와 판정을 CSV 로 낸다.
    evaluate   라벨과 대조해 성능(F1 등)을 잰다. 성능 요약 CSV 를 낸다.

    python -m lift_guard predict  --data data/prompts.csv --out outputs/
    python -m lift_guard evaluate --data data/benchmark.csv --out outputs/

입력·출력 형식은 README 의 '데이터 형식' 절에 정리돼 있다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .calibrate import best_threshold, metrics
from .guard import LiftGuard

csv.field_size_limit(10 ** 9)

#: 아무것도 지정하지 않았을 때 결과가 떨어지는 곳.
DEFAULT_OUT = Path("outputs")


def _read(path: Path, required: set[str]) -> list[dict]:
    if not path.is_file():
        sys.exit(f"입력 파일이 없다: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        sys.exit(f"빈 파일이다: {path}")
    if miss := required - set(rows[0]):
        sys.exit(f"{path.name} 에 열이 없다: {', '.join(sorted(miss))}\n"
                 f"  필요한 열: {', '.join(sorted(required))}")
    if bad := [i for i, r in enumerate(rows) if not (r.get("prompt") or "").strip()]:
        sys.exit(f"{path.name}[{bad[0]}]: prompt 가 비어 있다 (총 {len(bad)}행)")
    return rows


def _build(args) -> LiftGuard:
    kw = {"window": max(args.window, 1), "batch_size": args.batch_size,
          "max_length": args.max_length}
    if args.model_id:
        kw["model_id"] = args.model_id
    g = LiftGuard(**kw)
    stride = int(g.window * g.stride_ratio)
    tag = "창 없음(원문만)" if args.window == 0 else f"창 {g.window} / stride {stride}"
    print(f"모델 {g.model_id} | {tag} | max_length {g.max_length} | "
          f"batch {g.batch_size} | device {g.device}")
    return g


def _score(guard: LiftGuard, texts: list[str], window: int) -> np.ndarray:
    return guard.score_raw(texts) if window == 0 else guard.score(texts)


def _uids(rows: list[dict]) -> list[str]:
    return [r.get("uid") or str(i) for i, r in enumerate(rows)]


# ------------------------------------------------------------------- predict
def cmd_predict(args) -> None:
    rows = _read(args.data, {"prompt"})
    guard = _build(args)
    texts = [r["prompt"] for r in rows]
    score = _score(guard, texts, args.window)

    if args.threshold is None:
        print("\n경고: --threshold 를 주지 않았다. 점수만 내고 판정은 비운다.\n"
              "  가드마다 점수 척도가 자릿수 단위로 달라 안전한 기본 임계값이 없다.\n"
              "  라벨이 있는 데이터로 `evaluate` 를 먼저 돌려 임계값을 얻을 것.")
    pred = None if args.threshold is None else (score >= args.threshold).astype(int)

    args.out.mkdir(parents=True, exist_ok=True)
    f = args.out / "predictions.csv"
    with f.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "score", "pred"])
        for uid, s, i in zip(_uids(rows), score, range(len(rows))):
            w.writerow([uid, f"{s:.6f}", "" if pred is None else int(pred[i])])
    n = "" if pred is None else f" | 악성 {int(pred.sum()):,} / 정상 {len(pred) - int(pred.sum()):,}"
    print(f"\n{len(rows):,}행{n}\n산출: {f}")


# ------------------------------------------------------------------ evaluate
def cmd_evaluate(args) -> None:
    rows = _read(args.data, {"prompt", "label"})
    has_split = "split" in rows[0]
    has_domain = "domain" in rows[0]
    if not has_split:
        print("경고: split 열이 없다. 같은 데이터에서 임계값을 고르고 그대로 재므로\n"
              "  test F1 이 낙관적으로 나온다. valid/test 를 나눠 쓸 것.")

    guard = _build(args)
    texts = [r["prompt"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    split = (np.array([r["split"] for r in rows]) if has_split
             else np.array(["valid"] * len(rows)))
    # domain 열이 있으면 'baseline' 을 일반 탈옥(GJ), 나머지를 도메인 위장(DAJ)으로 본다.
    if has_domain:
        daj = np.array([r["domain"] != "baseline" for r in rows])
        cases = [("GJ", ~daj), ("DAJ", daj)]
    else:
        cases = [("all", np.ones(len(rows), dtype=bool))]

    windows = args.sweep or [args.window]
    out: list[dict] = []
    head = (f"{'window':>7s} {'case':5s} {'thr':>10s} | {'valid F1':>8s} | "
            f"{'test F1':>8s} {'P':>6s} {'R':>6s} | {'공격':>7s} {'정상':>7s}")
    print("\n" + head)
    print("-" * len(head))
    for w in windows:
        score = _score(guard if w == guard.window else _rewindow(guard, w), texts, w)
        for case, keep in cases:
            va = keep & (split == "valid")
            te = keep & (split == "test") if has_split else va
            if not va.any():
                print(f"{w:>7d} {case:5s} valid 표본이 없다"); continue
            t, f1_va = best_threshold(score[va], y[va])
            m = metrics(score[te], y[te], t)
            print(f"{'none' if w == 0 else w:>7} {case:5s} {t:10.6f} | {f1_va:8.3f} | "
                  f"{m['f1']:8.3f} {m['precision']:6.3f} {m['recall']:6.3f} | "
                  f"{int((te & (y == 1)).sum()):7,} {int((te & (y == 0)).sum()):7,}")
            out.append({
                "window": "none" if w == 0 else w,
                "stride": "" if w == 0 else int(w * guard.stride_ratio),
                "case": case, "threshold": round(t, 6),
                "valid_f1": round(f1_va, 4), "test_f1": round(m["f1"], 4),
                "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                "n_test_attack": int((te & (y == 1)).sum()),
                "n_test_benign": int((te & (y == 0)).sum()),
            })

    args.out.mkdir(parents=True, exist_ok=True)
    f = args.out / "metrics.csv"
    with f.open("w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"\n산출: {f}")


def _rewindow(guard: LiftGuard, window: int) -> LiftGuard:
    """이미 올린 모델을 그대로 두고 창 크기만 바꾼다(sweep 용)."""
    guard.window = max(window, 1)
    return guard


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lift_guard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", type=Path, required=True,
                       help="입력 CSV 경로")
        p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                       help=f"결과를 저장할 디렉터리 (기본 {DEFAULT_OUT}/)")
        p.add_argument("--window", type=int, default=64,
                       help="창 크기. 0 이면 창 분할 없음. stride 는 창의 50%%.")
        p.add_argument("--model-id", default=None, help="다른 가드를 쓸 때")
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--max-length", type=int, default=None,
                       help="한 번의 순전파에서 자를 길이. 기본은 모델 컨텍스트 길이.")

    p = sub.add_parser("predict", help="판정만 한다(라벨 불필요)")
    common(p)
    p.add_argument("--threshold", type=float, default=None,
                   help="악성으로 볼 점수 하한. 생략하면 점수만 내고 판정은 비운다.")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="라벨과 대조해 성능을 잰다")
    common(p)
    p.add_argument("--sweep", nargs="*", type=int, default=None,
                   help="여러 창 크기를 한 번에 잰다. 예: --sweep 0 8 16 32 64 128")
    p.set_defaults(func=cmd_evaluate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
