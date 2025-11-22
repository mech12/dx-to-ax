# Jupyter Notebook 환경 설정 가이드

## 1. myenv 가상환경 생성 및 설정

### 1.1 가상환경 생성
```bash
python -m venv myenv
```

### 1.2 필요한 패키지 설치
```bash
source myenv/bin/activate
pip install ipykernel jupyter numpy pandas scikit-learn matplotlib seaborn
```

### 1.3 Jupyter 커널로 등록
```bash
source myenv/bin/activate
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
```

등록 완료 메시지: `Installed kernelspec myenv in /Users/roy/Library/Jupyter/kernels/myenv`

## 2. Jupyter Notebook 실행

### 2.1 기본 실행 방법
```bash
source myenv/bin/activate
jupyter notebook
```

### 2.2 백그라운드 실행
```bash
source myenv/bin/activate
jupyter notebook &
```

### 2.3 접속 정보
- **서버 주소**: http://localhost:8889 (포트 번호는 자동으로 할당될 수 있음)
- **토큰 포함 URL**: 실행 시 출력되는 URL 사용
  ```
  http://localhost:8889/tree?token=<YOUR_TOKEN>
  ```

## 3. Jupyter Notebook에서 myenv 커널 사용하기

1. 브라우저에서 Jupyter Notebook 접속
2. 노트북 파일(`.ipynb`) 열기
3. 상단 메뉴: **Kernel > Change Kernel > Python (myenv)** 선택
4. 셀 실행하여 코드 테스트

## 4. 설치된 패키지 목록

myenv 환경에 설치된 주요 패키지:

- **ipykernel**: Jupyter 커널 지원
- **jupyter**: Jupyter Notebook 및 JupyterLab
- **numpy**: 수치 계산 라이브러리
- **pandas**: 데이터 분석 라이브러리
- **scikit-learn**: 머신러닝 라이브러리
- **matplotlib**: 데이터 시각화 라이브러리
- **seaborn**: 통계 데이터 시각화 라이브러리

## 5. 커널 관리

### 5.1 설치된 커널 목록 확인
```bash
jupyter kernelspec list
```

### 5.2 커널 제거
```bash
jupyter kernelspec uninstall myenv
```

### 5.3 커널 재등록 (필요시)
```bash
source myenv/bin/activate
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
```

## 6. Jupyter Notebook 종료

### 6.1 일반 종료
터미널에서 `Ctrl+C` 두 번 누르기

### 6.2 백그라운드 프로세스 종료
```bash
# 프로세스 ID 확인
ps aux | grep jupyter

# 프로세스 종료
kill <PID>
```

## 7. 프로젝트 파일

현재 프로젝트에서 사용 가능한 노트북 파일:

- **scikit-learn-iris.ipynb**: Iris 데이터셋 분석 예제
  - load_iris() 함수 사용
  - 로지스틱 회귀 및 의사결정나무 모델
  - 데이터 시각화 포함

## 8. 문제 해결

### 8.1 포트가 이미 사용 중인 경우
Jupyter는 자동으로 다른 포트를 찾습니다 (8888 → 8889 → 8890 ...)

특정 포트 지정:
```bash
jupyter notebook --port 9999
```

### 8.2 커널을 찾을 수 없는 경우
```bash
# 가상환경 활성화 확인
which python

# 커널 재등록
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
```

### 8.3 패키지 import 오류
```bash
# 가상환경에서 패키지 재설치
source myenv/bin/activate
pip install --upgrade <package-name>
```

## 9. 빠른 시작 가이드

```bash
# 1. 가상환경 활성화
source myenv/bin/activate

# 2. Jupyter Notebook 실행
jupyter notebook

# 3. 브라우저에서 자동으로 열리는 URL 접속
# 또는 터미널에 출력된 URL을 복사하여 브라우저에 붙여넣기

# 4. scikit-learn-iris.ipynb 파일 열기

# 5. Kernel > Change Kernel > Python (myenv) 선택

# 6. 셀 실행 (Shift + Enter)
```

## 10. 추가 팁

- **셀 실행**: `Shift + Enter`
- **새 셀 추가 (아래)**: `B` (명령 모드)
- **새 셀 추가 (위)**: `A` (명령 모드)
- **셀 삭제**: `DD` (명령 모드)
- **편집 모드**: `Enter`
- **명령 모드**: `Esc`
- **모든 셀 실행**: `Kernel > Restart & Run All`
