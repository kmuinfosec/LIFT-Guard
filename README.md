<h1 align="center">🛡️ LIFT-Guard</h1>

<p align="center">
  <b>Context-Specific Jailbreak Attacks and Defense for Commercial LLM Services</b><br>
  <sub>상용 LLM 서비스를 위한 맥락 특화형 탈옥 공격 및 방어 (CISC-W 2026)</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/🤗_Transformers-FFD21E">
  <img src="https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

**LIFT-Guard**(**L**ocal **I**nspection with **F**ull-prompt **T**racking)는 업무 맥락으로 위장한
탈옥 공격(Context-Specific Jailbreak, **CSJ**)을 탐지하는 가드 모델 래퍼입니다.
입력을 토큰 단위 슬라이딩 윈도우로 잘라 각 청크와 원문 전체를 경량 인코더 가드에 넣고,
**위험 점수의 최댓값**으로 판정합니다. 재학습 없이 입력 전처리만으로 동작합니다.


| 🎯 **F1 0.940**          | CSJ 탐지에서 기존 생성형 가드 6종(최대 8B)을 모두 능가        |
| ------------------------ | ------------------------------------------ |
| 🪶 **86M 파라미터**          | WildGuard 7B의 1/81, Qwen3Guard 8B의 1/93 크기 |
| 🔒 **ASR 15.5% → 0.17%** | Gemma-4-31B 대상 CSJ 공격 성공률 (방어 전 → 적용 후)    |
| ⚡ **학습 불필요**             | Prompt Guard 2 86M을 변형 없이 그대로 사용           |

---

## 🔍 Overview

금융·보험·여행 등 도메인 특화 LLM 서비스의 **업무 맥락은 탈옥 공격에 악용**될 수 있습니다.
일반 탈옥 프롬프트는 차단되지만, 같은 유해 요청을 서비스의 정상 업무처럼 포장한 CSJ는
정렬을 쉽게 우회합니다. 본 연구는 서비스 제공자의 승인 하에 **실제 운영 중인 국내 상용 LLM
서비스**에서 이를 실증했습니다.

<p align="center"><img src="assets/fig1_csj_example.jpg" width="85%"></p>

공개 LLM 대상 확장 실험에서도 CSJ는 일반 탈옥(JBB)과 전략 특화형 탈옥(SJ)보다 일관되게 높은
공격 성공률(ASR)을 보입니다.

<p align="center"><img src="assets/fig2_asr.jpg" width="60%"></p>

**기존 가드가 CSJ를 놓치는 이유** — 유해한 부분은 전체 프롬프트의 일부에 불과하므로,
프롬프트를 한 덩어리로 판별하면 공격 신호가 정상 문맥에 희석됩니다.

| 가드 모델 | SJ | CSJ | ΔF1 |
|---|:-:|:-:|:-:|
| Granite Guardian 3.0 2B | 0.958 | 0.804 | **−0.154** |
| Qwen3Guard 0.6B | 0.989 | 0.887 | −0.102 |
| Llama Guard 3 1B | 0.971 | 0.883 | −0.088 |
| Qwen3Guard 8B | 0.997 | 0.925 | −0.072 |

---

## ⚙️ Method

<p align="center"><img src="assets/fig3_overview.jpg" width="100%"></p>

1. 입력 질의를 토큰화한다.
2. 윈도우 크기 `W`, 중첩 `S = W/2`의 토큰 단위 슬라이딩 윈도우로 청크 `C1 … Cn`을 만든다.
3. 각 청크와 원문 전체를 경량 인코더 가드에 **독립적으로** 넣어 위험 점수를 얻는다.
4. 최댓값을 최종 점수로 삼고, 임계값 `τ` 이상이면 차단한다.

```
score(prompt) = max( guard(prompt), max_i guard(window_i) )
```

청크로 나누면 악의적 의도가 업무 맥락과 분리되어 개별 청크에서 드러나고, 청크를 중첩시켜
경계에서 신호가 잘리는 것을 막으며, 원문 점수를 함께 써서 기존 가드의 탐지력도 유지합니다.
청크는 배치로 병렬 처리되므로 7~8B 생성형 가드보다 추론 비용이 낮습니다.

---

## 🏆 Results

### CSJ 탐지 성능 

<p align="center"><img src="assets/fig4_f1.jpg" width="55%"></p>

| 가드 모델 | 파라미터 | F1 |
|---|:-:|:-:|
| **LIFT-Guard** (Prompt Guard 2, W=64) | **86M** | **0.940** |
| Qwen3Guard 8B | 8B | 0.925 |
| WildGuard 7B | 7B | 0.916 |
| Prompt Guard 2 (분할 없음) | 86M | 0.914 |
| Qwen3Guard 4B | 4B | 0.900 |
| Qwen3Guard 0.6B | 0.6B | 0.887 |
| Llama Guard 3 1B | 1B | 0.883 |
| Granite Guardian 3.0 2B | 2B | 0.804 |

### 창 크기별 성능

| 창 크기 | 8 | 16 | 32 | **64** | 128 | 256 | 분할 없음 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| F1 (CSJ) | 0.896 | 0.914 | 0.933 | **0.940** | 0.925 | 0.914 | 0.914 |
| F1 (SJ) | 0.885 | 0.913 | 0.956 | **0.980** | 0.972 | 0.972 | 0.972 |

창이 작으면(8, 16) 청크가 문장 하나를 담지 못해 유해 의도가 경계에서 끊기고, 창이 크면
(128, 256) 도메인 문구가 다시 포함되어 희석이 발생합니다. 기본값은 **64**입니다.

<details>
<summary>실험 설정</summary>

* **데이터**: 공격 질의 CSJ 3,500건 + SJ 700건, 정상 질의 업무 관련 3,500건 + 일반 700건.
  valid:test = 5:5로 나누고 valid에서 F1을 최대화하는 임계값을 선택한 뒤 test에서 측정.
* **공격 생성**: JBB 유해 프롬프트 100건 × 5개 도메인 시스템 프롬프트(여행·보험·B2B 영업·주식 분석·탐사 보도)
  → 공격자 LLM(Qwen-3.8-27B)으로 SJ/CSJ 생성. GOAT 전략 7종 적용.
* **비교 모델**: 기존 가드 6종, 각 모델 기본 설정.
</details>

---

## 🚀 Quick Start

```bash
uv sync
```

```python
from lift_guard import LiftGuard

guard = LiftGuard(window=64)                 # 보폭은 창 크기의 50%
guard.calibrate(val_prompts, val_labels)     # 임계값
guard.predict(["...prompt..."])              # -> bool 배열
guard.score(["...prompt..."])                # -> 연속 점수
```

```bash
uv run examples/quickstart.py
```

### CLI

| 모드 | 기능 | 라벨 | 출력 |
|---|---|:-:|---|
| `predict` | 프롬프트를 채점하고 판정한다 | 불필요 | `<out>/predictions.csv` |
| `evaluate` | 라벨과 대조하여 성능을 측정한다 | 필요 | `<out>/metrics.csv` |

```bash
# 성능 측정: valid 분할에서 임계값을 정하고 test 분할에 적용한다
uv run python -m lift_guard evaluate --data data/benchmark.csv --out outputs/

# 여러 창 크기를 한 번에 평가한다 (0은 분할 없음)
uv run python -m lift_guard evaluate --data data/benchmark.csv --out outputs/ \
    --sweep 0 8 16 32 64 128 256

# 판정만 수행한다. evaluate에서 얻은 임계값을 지정한다
uv run python -m lift_guard predict --data data/prompts.csv --out outputs/ \
    --threshold 0.001480
```

공통 옵션: `--window`(기본 64, 0이면 분할 없음), `--model-id`, `--batch-size`(기본 64),
`--max-length`(기본은 모델의 최대 컨텍스트 길이).
