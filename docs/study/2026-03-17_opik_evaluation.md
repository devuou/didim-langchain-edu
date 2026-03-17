# Opik Evaluation 가이드

> 출처: Opik 공식 문서

---

## 1. Overview — 평가 기능 개요

Opik의 평가 시스템은 LLM 앱의 출력 품질을 체계적으로 측정하고 반복 개선하기 위한 프레임워크입니다.

### 평가 3대 구성 요소

| 구성 요소 | 설명 |
|---|---|
| **Dataset** | 테스트할 입력과 기대 출력 모음 |
| **Evaluation Task** | 데이터셋 입력을 받아 LLM 출력을 생성하는 함수 |
| **Metrics** | 출력 품질을 채점하는 지표 (Hallucination, AnswerRelevance 등) |

### 평가 방법 2가지

| 메서드 | 용도 |
|---|---|
| `evaluate_prompt` | 단순 프롬프트 평가 |
| `evaluate` | 복잡한 에이전트/체인 평가 (우리 케이스) |

---

## 2. Concepts — 핵심 개념

### Dataset (데이터셋)
평가의 기초가 되는 테스트 케이스 모음. 생성 방법 3가지:
- 직접 예시 작성
- 합성 데이터 생성 도구 활용
- 프로덕션 트레이스에서 변환

### Experiment (실험)
평가를 한 번 실행할 때마다 Experiment 하나 생성.

- **Experiment Configuration**: 사용한 프롬프트, 모델, 파라미터 등 메타데이터
- **Experiment Items**: 각 데이터셋 항목별 입력/기대 출력/실제 출력/점수 + 트레이스 연결

### Multi-Value Feedback Scores
여러 팀원이 동일 항목에 각자 점수를 매길 수 있고, UI에서 평균·표준편차 등 통계 자동 제공.

---

## 3. Manage Datasets — 데이터셋 관리

### 생성 방법 (4가지)
1. UI에서 직접 생성 (CSV 업로드, 최대 1,000행)
2. Python/TypeScript SDK로 생성
3. 프로덕션 트레이스에서 바로 추가
4. JSONL 파일 또는 Pandas DataFrame에서 업로드

### Dataset 항목 SDK 코드

```python
from opik import Opik
import pandas as pd

client = Opik()
dataset = client.get_or_create_dataset(name="My dataset")

# 방법 1: 딕셔너리 리스트로 직접 삽입
dataset.insert([
    {"input": "AAPL 현재가 알려줘", "expected_output": "AAPL 현재가: $..."},
    {"input": "TSLA 52주 최고가는?", "expected_output": "TSLA 52주 최고가: $..."},
])

# 방법 2: JSONL 파일에서 적재
dataset.read_jsonl_from_file("path/to/file.jsonl")

# 방법 3: Pandas DataFrame에서 적재
df = pd.DataFrame({...})
dataset.insert_from_pandas(
    dataframe=df,
    keys_mapping={"컬럼명": "필드명"}
)
```

- `input`, `expected_output` 키 이름 **권장** (관례)
- 임의의 키-값도 허용 — evaluation task에서 `item["키명"]`으로 접근
- 삽입 시 중복 항목 자동 제거
- 수정 시마다 자동으로 새 버전(v1, v2…) 생성, 각 Experiment는 실행 당시 버전과 영구 연결

### AI로 데이터셋 확장 (Expand with AI)
기존 5~10개 데이터만 있어도 AI가 유사하지만 다양한 테스트 케이스 자동 생성 → Draft 상태로 저장 후 검토

### 태그와 필터링
`production`, `edge-case`, `needs-review` 등 태그를 붙여 특정 부분집합만 골라 실험 실행 가능.

---

## 4. Metrics — 전체 목록 및 입력 필드 명세

### Heuristic Metrics (규칙 기반, 결정론적)

| Metric | 필수 필드 | 선택 필드 |
|---|---|---|
| `Equals` | `output`, `reference` | — |
| `Contains` | `output`, `reference` | `case_sensitive` |
| `RegexMatch` | `output` | — |
| `IsJson` | `output` | — |
| `LevenshteinRatio` | `output`, `reference` | — |
| `SentenceBLEU` | `output`, `reference` | `n_grams` 등 |
| `ROUGE` | `output`, `reference` | `rouge_type` |
| `BERTScore` | `output`, `reference` | `model_type` |
| `Sentiment` | `output` | — |
| `Readability` | `output` | `min_grade`, `max_grade` |

### LLM-as-a-Judge Metrics

> evaluation task 반환 딕셔너리의 키가 각 metric의 `score()` 파라미터명과 **정확히 일치**해야 함.
> 불일치 시 `scoring_key_mapping`으로 매핑.

| Metric | 필수 필드 | 선택 필드 |
|---|---|---|
| `Hallucination` | `input`, `output` | `context` (list) |
| `AnswerRelevance` | `input`, `output`, `context` | `require_context=False`로 context 선택화 가능 |
| `ContextPrecision` | `input`, `output`, `expected_output`, `context` (list) | — |
| `ContextRecall` | `input`, `output`, `expected_output`, `context` (list) | — |
| `Moderation` | `output` | — |
| `Summarization` | `input`, `output` | — |
| `AgentTaskCompletionJudge` | `input`, `output` | — |
| `Usefulness` | `input`, `output` | — |
| `GEval` | metric 정의 시 `criteria`/`task`에 따라 동적 결정 | — |

---

## 5. Evaluate Your Agent — 에이전트 평가 실행 (5단계)

### Step 1: 트래킹 추가 (선택)
`@track` 데코레이터나 `track_langgraph`로 트레이싱을 붙이면 평가 결과와 실행 흐름을 함께 분석 가능.
→ **우리 에이전트는 이미 `track_langgraph` 연동 완료**

### Step 2: Evaluation Task 정의
데이터셋 항목을 입력으로 받아 LLM 출력을 딕셔너리로 반환하는 **동기 함수**.

```python
def evaluation_task(item):
    result = your_llm_application(item["input"])
    return {"output": result}
```

### Step 3: Dataset 선택

```python
dataset = opik_client.get_or_create_dataset("dataset-name")
```

### Step 4: Metric 선택

| 종류 | 예시 | 특징 |
|---|---|---|
| **Heuristic metrics** | `Equals`, `Contains` | 결정론적, 빠름 |
| **LLM-as-a-Judge** | `Hallucination`, `AnswerRelevance` | LLM이 직접 품질 판단 |

### Step 5: `evaluate()` 전체 파라미터

```python
from opik.evaluation import evaluate

evaluate(
    dataset,                      # [필수] Opik Dataset 객체
    task,                         # [필수] 동기 evaluation task 함수
    scoring_metrics,              # [필수] Metric 목록
    experiment_name,              # [선택] Opik UI에 표시될 실험 이름
    experiment_config,            # [선택] 실험 메타데이터 dict (모델명, 버전 등)
    project_name,                 # [선택] trace를 기록할 Opik 프로젝트 이름
    nb_samples,                   # [선택] 평가할 샘플 수 제한
    dataset_filter_string,        # [선택] OQL 문법으로 항목 필터링
    scoring_key_mapping,          # [선택] task 반환 키 → metric 파라미터명 매핑
    task_threads,                 # [선택] 병렬 실행 스레드 수
    scoring_functions,            # [선택] 커스텀 스코링 함수 목록
    experiment_scoring_functions, # [선택] 실험 전체 레벨 집계 메트릭
    prompts,                      # [선택] 연결할 Opik Prompt 버전 목록
)
```

---

## 6. LLM-as-a-Judge 채점 모델 설정

기본 채점 모델: **`gpt-4o-mini`** (LiteLLM 경유)

### 모델 교체 방법

```python
# 방법 1: 문자열 직접 지정 (LiteLLM 형식)
metric = Hallucination(model="gpt-4o")

# 방법 2: LiteLLMChatModel 클래스 사용
from opik.evaluation import models
model = models.LiteLLMChatModel(model_name="gpt-4o-mini")
metric = Hallucination(model=model)

# 방법 3: OpenAI 호환 API (다른 프로바이더)
import os
os.environ["OPENAI_BASE_URL"] = "https://api.your-provider.com/v1"
metric = Hallucination(model="openai/your-model-name")
```

---

## 7. 고급 기능 요약

| 기능 | 설명 |
|---|---|
| 커스텀 Scoring 함수 | 직접 채점 로직 작성 가능 |
| Task Span Metric | 실행 시간, 스팬 계층 구조 등 내부 동작까지 평가 |
| Experiment-level Metric | 전체 테스트 결과에 대한 집계 지표 계산 (max, mean 등) |
| 데이터셋 필터링 | 특정 태그/조건의 항목만 골라 평가 |
| 프롬프트 버전 연결 | 실험을 특정 프롬프트 버전과 연결해 관리 |

---

## 8. 우리 프로젝트 적용 계획

### 우리 에이전트에 적합한 Metric

| Metric | 이유 |
|---|---|
| `Hallucination` | 없는 주가·수치를 지어냈는지 탐지 |
| `AnswerRelevance` | 주식 질문에 관련 있는 답변인지 확인 |
| `AgentTaskCompletionJudge` | 요청한 작업(주가 조회 등)을 실제로 완료했는지 |

### 디렉토리 구조 계획

```
evaluation/
├── metrics/
│   ├── stock_hallucination.py      # Hallucination 래핑 (도메인 특화)
│   ├── stock_answer_relevance.py   # AnswerRelevance 래핑 (도메인 특화)
│   └── stock_task_completion.py    # AgentTaskCompletionJudge 래핑 (도메인 특화)
└── run_eval.py                     # 데이터셋 정의 + evaluate() 실행 진입점
```

- 각 Metric 파일: `__init__()` + `score()` 메서드를 가진 클래스, 내부적으로 Opik 내장 Metric 호출
- `run_eval.py`: `uv run python evaluation/run_eval.py`로 독립 실행
