# Reasoning Workflows

언어모델의 답안 생성, 전체 답안 수정, 단계별 수정, 오류 탐지를 수행하는 도구입니다. LLM_reasoning과 self-correct의 공통 데이터 처리를 합쳤습니다.

## 코드 구조

| 경로 | 역할 |
|---|---|
| `reasoning_workflows/api/` | API 클라이언트를 사용하는 생성·수정·오류 탐지 |
| `reasoning_workflows/langchain/` | LangChain 기반 생성·평가 및 Gemini 실행 |
| `reasoning_workflows/data/` | 공통 데이터 변환·샘플링·답안 파싱 |
| `reasoning_workflows/models/` | OpenAI·Gemini 모델 어댑터 |
| `reasoning_workflows/dict_messages.py` | dict 메시지 직렬화 |
| `reasoning_workflows/configuration.py` | 외부 실행 파일 로드 |

## 실행

```bash
pip install -e ".[runtime]"
python -m reasoning_workflows.api.revise_whole_answer --config /path/to/execution.yaml
python -m reasoning_workflows.api.revise_reasoning_steps --config /path/to/execution.yaml
```

모델 클라이언트별 메시지 타입과 생성 방식이 달라 API판과 LangChain판을 분리했습니다. 두 버전은 data 패키지를 공유합니다. 실행 파일의 `model`, `template`, `test_path`, `out_path`, `max_examples` 및 workflow별 옵션을 읽습니다. 모델 토큰은 실행 환경에서 주입한 값으로 구성합니다. Gemini 직접 호출은 GOOGLE_API_KEY 환경변수를 사용합니다.

## Source and license

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for project origins and third-party licenses.
