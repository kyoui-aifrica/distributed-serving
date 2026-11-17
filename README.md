# H100 2장 병렬화 기초 실습

## 1. 목표

- [분산 서빙 요약](./references/SUMMARY.md)의 병렬화 기법(DP, TP, PP, EP)을 H100 80GB 2장에서 vLLM으로 띄워 주요 지표 비교
- 비교 지표는 TTFT, ITL, 출력 처리량
- 하이퍼스케일(TP + PP)은 서버 여러 대, PD 분리는 별도 구성이 필요해 제외
- 전체 소요 시간 약 1.5시간 (구성 8개, 서버 기동 포함한 추정)

| 실험 | 질문 | 모델 |
| --- | --- | --- |
| 1 | 1장으로 서빙할 수 있는 모델에 2장을 쓰면 기법별로 지표가 어떻게 달라지나 | Qwen3-14B |
| 2 | 1장에 들어가지 않아 2장으로 서빙해야 하는 모델은 TP와 PP 중 무엇이 나은가 | Qwen2.5-72B-Instruct |
| 3 | MoE 모델은 TP와 EP 중 어느 쪽으로 나누나 | Qwen3-30B-A3B |

## 2. 사전 준비

### 2.1. vLLM 설치

```bash
UV_HTTP_TIMEOUT=300 uv sync
source .venv/bin/activate
```

- `pyproject.toml`과 `uv.lock`에 고정한 버전(vLLM 0.34.0, torch 2.14.0 CUDA 13.0 빌드)을 가상 환경(`.venv`)에 설치
- 이후 모든 명령은 `source .venv/bin/activate` 후 실행
- `UV_HTTP_TIMEOUT=300`은 네트워크가 느릴 때 큰 파일 다운로드가 끊기지 않게 대기 시간을 늘림

### 2.2. 모델 다운로드

```bash
hf download Qwen/Qwen3-14B
hf download Qwen/Qwen2.5-72B-Instruct
hf download Qwen/Qwen3-30B-A3B
```

| 모델 | 종류 | 크기 |
| --- | --- | --- |
| `Qwen/Qwen3-14B` | dense, BF16 | 약 30 GB |
| `Qwen/Qwen2.5-72B-Instruct` | dense, BF16 | 약 145 GB |
| `Qwen/Qwen3-30B-A3B` | MoE (전문가 128개 중 토큰마다 8개 사용), BF16 | 약 61 GB |

- 합계 약 236 GB, 기본 저장 위치는 `~/.cache/huggingface`
- 미리 받지 않으면 `vllm serve`가 시작하면서 내려받아, 서버 기동 대기 시간(30분)을 넘길 수 있음

### 2.3. 환경 기록

```bash
nvidia-smi
nvidia-smi topo -m
vllm --version
python -c "import torch; print(torch.__version__)"
```

- `nvidia-smi topo -m`에서 GPU0과 GPU1이 만나는 칸이 `NV#`이면 NVLink 연결, 그 외는 PCIe 연결
- TP와 EP는 레이어마다 GPU 간 통신이 있어서 연결 방식에 따라 결과가 달라짐
- 출력은 `experiments/RESULTS.md` 1절에 기록

## 3. 측정 지표

| 지표 | 결과 키 | 뜻 |
| --- | --- | --- |
| TTFT | `median_ttft_ms` | 요청부터 첫 토큰까지 시간 (첫 토큰을 받은 시각 − 요청을 보낸 시각) |
| ITL | `median_itl_ms` | 출력 토큰 사이 간격 (토큰을 받은 시각 − 바로 앞 토큰을 받은 시각) |
| 출력 처리량 | `output_throughput` | 초당 생성한 출력 토큰 수 (모든 요청의 출력 토큰 합 ÷ 첫 요청부터 마지막 요청이 끝날 때까지 걸린 시간) |

- 프리필은 연산 성능, 디코드는 메모리 대역폭이 속도를 결정 ([요약 3.1절](references/SUMMARY.md#31-프리필과-디코드의-병목))
- p99 값(`p99_ttft_ms`, `p99_itl_ms`)으로 가장 느린 1% 요청 확인

## 4. 실험 구성

### 4.1. 공통 조건

| 항목 | 값 |
| --- | --- |
| 서버 옵션 | `--max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching` |
| 요청 | 입력 1024 토큰, 출력 256 토큰, 무작위 토큰 |
| 동시 요청 수 | 1, 8, 32, 128 (실험 2는 1, 8, 16) |

- 동시 요청 1은 지연(TTFT, ITL), 동시 요청 128은 처리량을 보는 지점
- 예상 결과의 ITL은 동시 요청 1, 출력 처리량은 동시 요청 128 기준이고, ITL은 짧을수록 빠름
- 프리픽스 캐시는 같은 입력을 재사용해 TTFT가 실제보다 낮게 나오지 않도록 끔

### 4.2. 실험 1. 1장으로 서빙할 수 있는 모델 (Qwen3-14B)

| ID | 기법 | 핵심 옵션 | 예상 결과 |
| --- | --- | --- | --- |
| `14b-1gpu` | 기준 | `CUDA_VISIBLE_DEVICES=0` | 비교 기준 |
| `14b-dp2` | [DP](references/SUMMARY.md#21-데이터-병렬화-dp) | `--data-parallel-size 2` | 출력 처리량이 1장보다 높음 |
| `14b-tp2` | [TP](references/SUMMARY.md#22-텐서-병렬화-tp) | `--tensor-parallel-size 2` | ITL이 1장보다 빠름 |
| `14b-pp2` | [PP](references/SUMMARY.md#23-파이프라인-병렬화-pp) | `--pipeline-parallel-size 2` | ITL이 TP보다 느림 |

- **DP**: 2장이 각자 요청을 받아 처리해 출력 처리량 증가
- **TP**: 두 GPU가 가중치를 절반씩 동시에 읽어 ITL 감소
- **PP**: 두 GPU가 가중치를 절반씩 차례로 읽어, 동시에 읽는 TP보다 ITL 증가

### 4.3. 실험 2. 2장으로 서빙해야 하는 모델 (Qwen2.5-72B-Instruct)

| ID | 기법 | 핵심 옵션 | 예상 결과 |
| --- | --- | --- | --- |
| `72b-tp2` | TP | `--tensor-parallel-size 2 --gpu-memory-utilization 0.95` | ITL이 PP보다 빠름 |
| `72b-pp2` | PP | `--pipeline-parallel-size 2 --gpu-memory-utilization 0.95` | ITL이 TP보다 느림 |

- **TP**: 두 GPU가 가중치를 절반씩 동시에 읽어 ITL 감소
- **PP**: 두 GPU가 가중치를 절반씩 차례로 읽어 ITL 증가

### 4.4. 실험 3. MoE 모델 (Qwen3-30B-A3B)

| ID | 기법 | 핵심 옵션 | 예상 결과 |
| --- | --- | --- | --- |
| `moe-tp2` | TP | `--tensor-parallel-size 2` | 비교 기준 |
| `moe-ep2` | [EP](references/SUMMARY.md#25-전문가-병렬화-ep) | `--data-parallel-size 2 --enable-expert-parallel` | 출력 처리량이 TP보다 높음 |

- **TP**: 어텐션과 전문가를 모두 반씩 쪼개, 레이어마다 두 GPU의 결과를 합치는 통신 발생
- **EP**: 어텐션은 GPU마다 다른 요청을 따로 처리하고 전문가만 GPU마다 64개씩 나눠 맡아, 어텐션 레이어의 통신이 없어 출력 처리량 증가

## 5. 실행

```bash
tmux new -s lab                                # SSH가 끊겨도 계속 실행, 다시 붙을 때는 tmux attach -t lab
cd experiments
./run_all.sh 2>&1 | tee run_all.log            # 전체 실행
```

- `run.sh <ID>`는 서버 시작, KV 캐시 토큰 수 기록, 동시 요청 수별 측정, 서버 종료 순서로 진행
- `run_all.sh`는 모든 구성을 순서대로 실행하고, 실패한 구성은 건너뛴 뒤 마지막에 성공과 실패 ID 출력
- 이미 있는 결과는 건너뜀, 구성 하나를 다시 측정하려면 `rm -r results/<ID>` 후 실행
- 실패 원인은 `results/<ID>/server.log`에서 확인

| 결과 파일 | 내용 |
| --- | --- |
| `results/<ID>/meta.json` | 구성 ID, 모델, GPU 수, 서버 옵션 |
| `results/<ID>/server.log` | `vllm serve` 전체 로그 |
| `results/<ID>/kv.txt` | 서버 로그에서 뽑은 KV 캐시 토큰 수, TTFT가 튈 때 요청이 대기열에서 밀렸는지 확인용 |
| `results/<ID>/in1024_out256_c<C>.json` | 동시 요청 C의 `vllm bench serve` 결과 |

## 6. 결과 보기

- `experiments/analysis.ipynb`를 열어 모든 셀 실행
- 표와 그래프를 보고 `experiments/RESULTS.md`에 실험별 결과 기록
