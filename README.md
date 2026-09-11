# LIFT-Guard

**L**ocal **I**nspection by **F**ixed-window **T**raversal — 도메인 위장 탈옥에서
가드 모델의 정확도를 되살리는 래퍼. **학습하지 않는다.**

## 문제

도메인 위장 탈옥은 유해 요청을 길고 그럴듯한 업무 프롬프트 안에 녹여 넣는다. 여행
일정표, 보험 갱신 안내문, 주식 리서치 노트 같은 것이다. 유해한 부분은 전체 텍스트의
일부에 불과하고, 프롬프트를 한 덩어리로 읽는 가드는 그 신호를 희석시켜 통과시킨다.

효과는 크고, 작은 가드일수록 크게 당한다. 같은 유해 요청을 도메인 문맥으로 감쌌을 때의
F1 손실이다.

| 가드 | 일반 탈옥 | 도메인 위장 | Δ F1 |
|---|---|---|---|
| Granite Guardian 3.0 2B | .958 | .804 | −.154 |
| Qwen3Guard 0.6B | .989 | .887 | −.102 |
| Llama Guard 3 1B | .971 | .883 | −.088 |
| Qwen3Guard 8B | .997 | .925 | −.072 |

## 방법

프롬프트를 겹치는 창으로 잘라 각각 채점하고, **원문 전체 점수와 함께** 최댓값을 취한다.

```
score(prompt) = max( guard(prompt), max_i guard(window_i) )
```

창 경계는 가드 자신의 토크나이저가 아니라 고정된 외부 토크나이저
(`all-MiniLM-L6-v2`)의 토큰 offset 으로 잡는다. 가드마다 토크나이저가 달라, 그렇게 하지
않으면 "64 토큰 창"이 모델마다 다른 길이를 뜻하게 된다.

**원문 항을 빼면 안 된다.** 창 하나에 담기지 않는 문맥이 있어야 유해하다고 읽히는
프롬프트가 있고, 실제로 우리가 시험한 모든 창 크기에서 조각만 쓴 점수는 결합 점수보다
낮았다.

학습은 없다. 가드는 공개된 그대로 쓴다. 정하는 값은 판정 임계값 하나뿐이다.

## 결과

Prompt Guard 2 86M 에 LIFT-Guard 를 씌워, 8B 까지의 가드 9종과 도메인 위장 탈옥에서
비교했다(test 분할, 공격 : 정상 = 1 : 1).

| 가드 | F1 |
|---|---|
| **LIFT-Guard (Prompt Guard 2 86M, 창 64)** | **.940** |
| Qwen3Guard 8B | .925 |
| WildGuard 7B | .916 |
| Prompt Guard 2 86M (창 없음) | .914 |
| Qwen3Guard 4B | .900 |
| Qwen3Guard 0.6B | .887 |
| Llama Guard 3 1B | .883 |
| Granite Guardian 3.0 2B | .804 |

86M 인코더가 8B 디코더를 앞선다. 파라미터 93배 차이인데, 바꾼 것은 프롬프트를 넣는
방식뿐이다.

### 창 크기

stride 는 항상 창의 50% 다.

| 창 | 8 | 16 | 32 | **64** | 128 | 256 | 미적용 |
|---|---|---|---|---|---|---|---|
| F1 (도메인 위장) | .896 | .914 | .933 | **.940** | .925 | .914 | .914 |
| F1 (일반) | .885 | .913 | .956 | **.980** | .972 | .972 | .972 |

곡선은 아래로 볼록하고, 양 끝이 나빠지는 이유가 서로 다르다. 창이 작으면(8·16) 조각이
문장 하나를 못 담아 유해 의도가 창 경계에서 끊기고, 동시에 정상 조각이 우연히 높은
점수를 받아 최적 임계값이 밀려 올라가면서 precision 이 무너진다. 창이 크면(128·256)
도메인 문구가 조각 안으로 다시 들어와 희석이 되살아나고, 256 에서는 창을 쓰지 않은 것과
수치까지 정확히 같아진다.

**기본값은 64.** 다만 F1 이 아니라 ASR(공격 성공률)을 줄이는 게 목표라면 32 가 낫다.
창 32 는 precision 을 내주고 recall 을 가져가는데, ASR 에는 recall 만 반영되기 때문이다.

## 설치

```bash
pip install -e .
```

`torch`, `transformers`, `numpy` 가 필요하다. 기본 가드
(`meta-llama/Llama-Prompt-Guard-2-86M`)는 gated 저장소라 모델 페이지에서 라이선스에
먼저 동의해야 한다. 다른 시퀀스 분류 가드를 쓰려면 `model_id` 로 넘기면 된다.

## 사용

두 가지 모드가 있다.

| 모드 | 하는 일 | 라벨 | 산출 |
|---|---|---|---|
| `predict` | 프롬프트를 채점하고 판정한다 | 불필요 | `<out>/predictions.csv` |
| `evaluate` | 라벨과 대조해 성능을 잰다 | 필요 | `<out>/metrics.csv` |

```bash
# 성능 측정 — valid 에서 임계값을 고르고 test 에 적용한다
python -m lift_guard evaluate --data data/benchmark.csv --out outputs/

# 창 크기를 한 번에 여러 개 (0 = 창 분할 없음)
python -m lift_guard evaluate --data data/benchmark.csv --out outputs/ \
    --sweep 0 8 16 32 64 128 256

# 판정만 — evaluate 가 알려준 임계값을 넣는다
python -m lift_guard predict --data data/prompts.csv --out outputs/ \
    --threshold 0.001480
```

공통 옵션: `--window`(기본 64, 0 이면 창 없음), `--model-id`, `--batch-size`(기본 64),
`--max-length`(기본은 모델 컨텍스트 길이).

파이썬에서 직접 쓸 수도 있다.

```python
from lift_guard import LiftGuard

guard = LiftGuard(window=64)                       # stride 는 창의 50%
guard.calibrate(val_prompts, val_labels)           # 임계값 하나만 정함, 학습 없음
guard.predict(["...prompt..."])                    # -> bool 배열
guard.score(["...prompt..."])                      # -> 연속 점수
```

```bash
python examples/quickstart.py
```

## 데이터 형식

입력은 **UTF-8 CSV** 하나다. 경로는 `--data` 로 준다.

### `predict` 입력

| 열 | 필수 | 내용 |
|---|---|---|
| `prompt` | 필수 | 가드에 넣을 텍스트 |
| `uid` | 선택 | 결과와 이어 붙일 식별자. 없으면 행 번호를 쓴다 |

```csv
uid,prompt
q-001,"Give me a 3-day vegetarian meal plan for a family of four."
q-002,"You are a travel planner. Build a 3-day Lisbon itinerary ... "
```

### `evaluate` 입력

`predict` 의 열에 다음을 더한다.

| 열 | 필수 | 내용 |
|---|---|---|
| `label` | 필수 | `1`=공격 / `0`=정상 |
| `split` | 권장 | `valid` \| `test`. valid 에서 임계값을 고르고 test 에서 잰다. 없으면 같은 데이터로 둘 다 해서 test F1 이 낙관적으로 나오고, 경고가 뜬다 |
| `domain` | 선택 | `baseline` 이면 일반 탈옥(GJ), 그 밖이면 도메인 위장(DAJ). 없으면 전체를 한 덩어리(`all`)로 잰다 |

```csv
uid,label,split,domain,prompt
atk-00001,1,valid,travel_planner,"You are a travel planner. ... "
ben-00001,0,test,baseline,"Explain how photosynthesis works to a 10 year old."
```

`domain` 으로 케이스를 가를 때는 **공격도 정상도 같은 기준으로 갈린다.** GJ 는
`domain=baseline` 인 공격과 정상, DAJ 는 도메인이 붙은 공격과 정상이다. 이렇게 해야 두
케이스 모두 양성:음성 비가 같아져 F1 을 나란히 비교할 수 있다. 정상을 두 케이스가
공유하면 비율이 달라져 비교가 깨진다.

## 산출 경로

`--out` 으로 지정한 디렉터리에 떨어진다. 생략하면 실행한 자리의 `outputs/` 다. 없으면
만든다.

### `outputs/predictions.csv`

| 열 | 내용 |
|---|---|
| `uid` | 입력의 `uid`(없으면 행 번호) |
| `score` | 유해 확률 [0,1] |
| `pred` | `1`=악성 / `0`=정상. `--threshold` 를 주지 않으면 비어 있다 |

### `outputs/metrics.csv`

케이스 × 창 크기마다 한 행이다.

| 열 | 내용 |
|---|---|
| `window`, `stride` | 창 설정. 창 없음은 `none` |
| `case` | `GJ` \| `DAJ` \| `all` |
| `threshold` | valid 에서 고른 값 |
| `valid_f1`, `test_f1` | 보정 분할과 평가 분할의 F1 |
| `precision`, `recall` | test 기준 |
| `tp`, `fp`, `fn`, `tn` | test 혼동행렬 |
| `n_test_attack`, `n_test_benign` | test 표본 수 |

## 임계값은 직접 보정할 것. 우리 값을 그대로 쓰지 말 것

가드마다 점수 척도가 자릿수 단위로 다르다. 우리 실험에서 F1 최적 임계값이 어떤 모델은
0.0015, 어떤 모델은 0.77 이었다. 0.5 같은 공통 기본값을 쓰면 그 척도에 맞지 않는 모델이
조용히 망가진다 — 0.5 에서 Prompt Guard 2 의 recall 은 0.099 였다.

실제 트래픽과 닮은 라벨 데이터로 보정하고, 트래픽에 성격이 다른 구간이 있으면 구간마다
따로 보정한다. `best_threshold` 는 분위수 격자가 아니라 **모든 고유 점수**를 훑으므로
최적값을 놓치지 않는다.

## 구현할 때 주의할 것

틀리면 정확도가 실제로 깎이는 두 가지다. 둘 다 우리가 겪었다.

- **패딩 방향.** 분류 헤드는 `hidden_states[:, 0]`, 즉 `[CLS]` 자리를 읽는다. 왼쪽
  패딩을 쓰면 그 자리가 PAD 토큰이 되고, 점수가 같은 배치에 무엇이 들어 있느냐에 따라
  달라진다. 우리 측정으로 최대 0.03 까지 흔들렸는데, 이 모델의 판정 임계값보다 한
  자릿수 큰 값이라 판정이 뒤집힌다. `LiftGuard` 는 오른쪽 패딩을 강제한다.
- **배치 구성.** 길이순으로 정렬해 묶지 않으면 패딩이 배치를 지배한다. 그렇게 해도
  fp16 에서는 배치 크기에 따라 점수가 1e-3 수준으로 움직이니, 비트 단위 재현이
  필요하면 배치 크기를 고정할 것.

## 라이선스

아직 정하지 않았다. 공개 전에 추가할 것. 가드 모델은 각자의 라이선스를 따르며
(Prompt Guard 2 는 Llama 라이선스), 이 래퍼가 그것을 바꾸지 않는다.
