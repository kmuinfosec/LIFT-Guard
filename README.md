# LIFT-Guard

LIFT-Guard(Local Inspection with Full-prompt Tracking)는 도메인 문맥으로 위장한 탈옥
공격(Context-strategied Jailbreak, CSJ)을 탐지하기 위한 가드 모델 래퍼이다. 입력을
일정 크기의 청크로 분할하여 각 청크와 원문 전체를 가드 모델에 넣고, 그중 최댓값을
최종 위협 점수로 사용한다. 가드 모델은 재학습하지 않으며, 입력 전처리만으로 동작한다.

## 배경

CSJ 공격은 유해 요청을 여행 일정표, 보험 안내문, 주식 리서치 노트와 같은 정상 업무
프롬프트 안에 녹여 넣는다. 유해한 부분은 전체 텍스트의 일부에 불과하므로, 프롬프트를
한 덩어리로 판별하는 가드 모델은 공격 신호가 정상 문맥에 희석되어 이를 통과시킨다.
같은 유해 요청을 도메인 문맥으로 감쌌을 때 기존 가드 모델의 F1은 다음과 같이
하락한다.

| 가드 모델 | SJ | CSJ | ΔF1 |
|---|---|---|---|
| Granite Guardian 3.0 2B | 0.958 | 0.804 | −0.154 |
| Qwen3Guard 0.6B | 0.989 | 0.887 | −0.102 |
| Llama Guard 3 1B | 0.971 | 0.883 | −0.088 |
| Qwen3Guard 8B | 0.997 | 0.925 | −0.072 |

SJ(Strategy-only Jailbreak)는 탈옥 전략만 적용한 공격, CSJ는 동일한 전략에 도메인
문맥을 결합한 공격이다.

## 방법

입력 프롬프트를 50% 중첩된 고정 크기 창으로 분할하여 각 창을 독립적으로 채점하고,
원문 전체의 점수와 함께 최댓값을 취한다.

```
score(prompt) = max( guard(prompt), max_i guard(window_i) )
```

## 실험 결과

Prompt Guard 2 86M에 LIFT-Guard를 적용하여 최대 8B 규모의 가드 모델과 CSJ 탐지 성능을
비교하였다(test 분할, 공격:정상 = 1:1).

| 가드 모델 | F1 |
|---|---|
| LIFT-Guard (Prompt Guard 2 86M, W=64) | 0.940 |
| Qwen3Guard 8B | 0.925 |
| WildGuard 7B | 0.916 |
| Prompt Guard 2 86M (분할 없음) | 0.914 |
| Qwen3Guard 4B | 0.900 |
| Qwen3Guard 0.6B | 0.887 |
| Llama Guard 3 1B | 0.883 |
| Granite Guardian 3.0 2B | 0.804 |

### 창 크기별 성능

보폭은 항상 창 크기의 50%이다.

| 창 크기 | 8 | 16 | 32 | 64 | 128 | 256 | 분할 없음 |
|---|---|---|---|---|---|---|---|
| F1 (CSJ) | 0.896 | 0.914 | 0.933 | 0.940 | 0.925 | 0.914 | 0.914 |
| F1 (SJ) | 0.885 | 0.913 | 0.956 | 0.980 | 0.972 | 0.972 | 0.972 |

창이 작으면(8, 16) 청크가 문장 하나를 담지 못해 유해 의도가 경계에서 끊기고, 정상
청크가 우연히 높은 점수를 받아 임계값이 올라가면서 precision이 떨어진다. 창이 크면
(128, 256) 도메인 문구가 청크 안에 다시 포함되어 희석이 발생하며, 256에서는 분할하지
않은 경우와 같은 값이 된다. 기본값은 64이다.

## 설치

```bash
pip install -e .
```

`torch`, `transformers`, `numpy`가 필요하다. 기본 가드 모델
(`meta-llama/Llama-Prompt-Guard-2-86M`)은 gated 저장소이므로 Hugging Face 모델
페이지에서 라이선스에 먼저 동의해야 한다. 다른 시퀀스 분류 모델을 사용하려면
`model_id`로 지정한다.

## 사용법

두 가지 모드를 제공한다.

| 모드 | 기능 | 라벨 | 출력 |
|---|---|---|---|
| `predict` | 프롬프트를 채점하고 판정한다 | 불필요 | `<out>/predictions.csv` |
| `evaluate` | 라벨과 대조하여 성능을 측정한다 | 필요 | `<out>/metrics.csv` |

```bash
# 성능 측정: valid 분할에서 임계값을 정하고 test 분할에 적용한다
python -m lift_guard evaluate --data data/benchmark.csv --out outputs/

# 여러 창 크기를 한 번에 평가한다 (0은 분할 없음)
python -m lift_guard evaluate --data data/benchmark.csv --out outputs/ \
    --sweep 0 8 16 32 64 128 256

# 판정만 수행한다. evaluate에서 얻은 임계값을 지정한다
python -m lift_guard predict --data data/prompts.csv --out outputs/ \
    --threshold 0.001480
```

공통 옵션: `--window`(기본 64, 0이면 분할 없음), `--model-id`, `--batch-size`(기본 64),
`--max-length`(기본은 모델의 최대 컨텍스트 길이).

Python에서 직접 호출할 수도 있다.

```python
from lift_guard import LiftGuard

guard = LiftGuard(window=64)                 # 보폭은 창 크기의 50%
guard.calibrate(val_prompts, val_labels)     # 임계값만 정한다. 학습은 없다
guard.predict(["...prompt..."])              # -> bool 배열
guard.score(["...prompt..."])                # -> 연속 점수
```

```bash
python examples/quickstart.py
```

## 데이터 형식

입력은 UTF-8 CSV 파일 하나이며, 경로는 `--data`로 지정한다.

### `predict` 입력

| 열 | 필수 | 내용 |
|---|---|---|
| `prompt` | 필수 | 가드 모델에 입력할 텍스트 |
| `uid` | 선택 | 결과와 연결할 식별자. 없으면 행 번호를 사용한다 |

```csv
uid,prompt
q-001,"Give me a 3-day vegetarian meal plan for a family of four."
q-002,"You are a travel planner. Build a 3-day Lisbon itinerary ... "
```

### `evaluate` 입력

`predict`의 열에 다음을 추가한다.

| 열 | 필수 | 내용 |
|---|---|---|
| `label` | 필수 | `1`=공격, `0`=정상 |
| `split` | 권장 | `valid` 또는 `test`. valid에서 임계값을 정하고 test에서 측정한다. 없으면 같은 데이터로 둘 다 수행하며, 이 경우 test F1이 낙관적으로 나오므로 경고를 출력한다 |
| `domain` | 선택 | `baseline`이면 SJ, 그 외 값이면 CSJ로 분류한다. 없으면 전체를 하나의 케이스(`all`)로 측정한다 |

```csv
uid,label,split,domain,prompt
atk-00001,1,valid,travel_planner,"You are a travel planner. ... "
ben-00001,0,test,baseline,"Explain how photosynthesis works to a 10 year old."
```

`domain`으로 케이스를 나눌 때는 공격과 정상 질의가 같은 기준으로 나뉜다. SJ는
`domain=baseline`인 공격과 정상 질의, CSJ는 도메인이 지정된 공격과 정상 질의로
구성된다. 이렇게 해야 두 케이스의 양성:음성 비율이 같아져 F1을 직접 비교할 수 있다.

## 출력

`--out`으로 지정한 디렉터리에 저장된다. 생략하면 현재 디렉터리의 `outputs/`이며,
없으면 생성한다.

### `outputs/predictions.csv`

| 열 | 내용 |
|---|---|
| `uid` | 입력의 `uid`(없으면 행 번호) |
| `score` | 유해 확률 [0, 1] |
| `pred` | `1`=악성, `0`=정상. `--threshold`를 지정하지 않으면 비어 있다 |

### `outputs/metrics.csv`

케이스 × 창 크기마다 한 행이 생성된다.

| 열 | 내용 |
|---|---|
| `window`, `stride` | 창 설정. 분할 없음은 `none` |
| `case` | `SJ`, `CSJ`, `all` 중 하나 |
| `threshold` | valid 분할에서 정한 값 |
| `valid_f1`, `test_f1` | 보정 분할과 평가 분할의 F1 |
| `precision`, `recall` | test 분할 기준 |
| `tp`, `fp`, `fn`, `tn` | test 분할의 혼동 행렬 |
| `n_test_attack`, `n_test_benign` | test 분할의 표본 수 |

## 라이선스

### 본 저장소

본 저장소의 코드는 MIT License를 따른다. 가드 모델의 가중치는 포함하거나 재배포하지
않으며, 아래 모델들의 라이선스를 변경하지 않는다.

```
MIT License

Copyright (c) 2026 [저작권자]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 사용 모델

각 모델은 Hugging Face에서 직접 내려받아야 하며, 해당 라이선스와 이용 정책에 사용자가
직접 동의해야 한다.

| 용도 | 모델 | 라이선스 | 비고 |
|---|---|---|---|
| 기본 가드 | `meta-llama/Llama-Prompt-Guard-2-86M` | [Llama 4 Community License](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M/blob/main/LICENSE) | gated. 배포물에 "Built with Llama" 표기와 Notice 파일이 필요하며 [Acceptable Use Policy](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M/blob/main/USE_POLICY.md)를 따라야 한다. 기반 모델 mDeBERTa-base는 MIT |
| 창 경계 토크나이저 | `sentence-transformers/all-MiniLM-L6-v2` | [Apache 2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 토크나이저 오프셋만 사용하며 임베딩 가중치는 로드하지 않는다 |
| 비교 실험 | `Qwen/Qwen3Guard-Gen-0.6B`, `-4B`, `-8B` | [Apache 2.0](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B) | |
| 비교 실험 | `allenai/wildguard` | [Apache 2.0](https://huggingface.co/allenai/wildguard) | gated. [AI2 Responsible Use Guidelines](https://allenai.org/responsible-use.pdf)에 동의하고 연구 목적 사용을 확인해야 한다 |
| 비교 실험 | `meta-llama/Llama-Guard-3-1B` | [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) | gated. Prompt Guard 2와 같은 종류의 조건이 적용된다 |
| 비교 실험 | `ibm-granite/granite-guardian-3.0-2b` | [Apache 2.0](https://huggingface.co/ibm-granite/granite-guardian-3.0-2b) | |

라이선스 정보는 2026년 9월 기준 각 모델 페이지에서 확인한 것이다. 배포 또는 상용
이용 전에는 해당 페이지의 원문을 다시 확인하기 바란다.
