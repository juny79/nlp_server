# Wandb 연동 가이드

## 1. Wandb 회원가입 및 API Key 발급

### 회원가입
1. https://wandb.ai 접속
2. 회원가입 (GitHub, Google 계정으로 가능)

### API Key 발급
1. 로그인 후 https://wandb.ai/authorize 접속
2. API Key 복사

## 2. 로컬 환경 설정

### 방법 1: 터미널에서 로그인 (권장)
```bash
# 가상환경 활성화
cd ~/nlp_server
source .venv/bin/activate

# wandb 로그인
wandb login
# API Key 입력 (붙여넣기)
```

### 방법 2: 환경변수로 설정
```bash
# .bashrc 또는 .zshrc에 추가
export WANDB_API_KEY="your_api_key_here"
```

### 방법 3: 코드에 직접 설정 (비권장 - 보안 위험)
```python
import os
os.environ["WANDB_API_KEY"] = "your_api_key_here"
```

## 3. Wandb 대시보드 설정

### 프로젝트 이름 변경 (선택사항)
`config.yaml` 파일에서 수정:
```yaml
wandb:
  enabled: true
  entity: null  # 팀이 있으면 팀명 입력
  project: dialogsum_solar  # 원하는 프로젝트명
  name: solar_v1_finetune   # 실험 이름
```

### 팀 프로젝트 사용 시
1. Wandb에서 팀 생성
2. `config.yaml`의 `entity`에 팀명 입력

## 4. 학습 시작

```bash
cd ~/nlp_server
source .venv/bin/activate
python train_solar_v1.py
```

## 5. Wandb 대시보드 확인

### 자동으로 기록되는 메트릭
- **Loss**: train_loss, eval_loss
- **Learning Rate**: learning_rate schedule
- **Epoch Progress**: epoch, step
- **GPU 사용률**: GPU memory, utilization
- **시스템 메트릭**: CPU, RAM 사용량

### 대시보드 접속
1. https://wandb.ai 로그인
2. 프로젝트 선택: `dialogsum_solar`
3. 실시간 학습 현황 확인

### 주요 기능
- **Charts**: Loss, Learning Rate 그래프
- **System**: GPU/CPU 사용률 모니터링
- **Logs**: 콘솔 출력 로그
- **Files**: 저장된 모델 체크포인트
- **Code**: 학습 코드 버전 관리

## 6. Wandb 끄기

학습 중 wandb를 끄고 싶다면 `config.yaml`에서:
```yaml
wandb:
  enabled: false
```

## 7. 유용한 Wandb 기능

### Sweeps (하이퍼파라미터 튜닝)
여러 하이퍼파라미터 조합을 자동으로 실험

### Artifacts
학습된 모델을 버전 관리하고 공유

### Reports
실험 결과를 보고서로 작성 및 공유

## 문제 해결

### "wandb: ERROR Error while calling W&B API"
- API Key가 올바른지 확인
- `wandb login` 재시도

### 로그인이 안 될 때
```bash
wandb login --relogin
```

### 오프라인 모드 사용
```bash
export WANDB_MODE=offline
```
학습 후 나중에 동기화:
```bash
wandb sync ./wandb/latest-run
```
