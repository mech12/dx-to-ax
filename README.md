# 2025-1108-ai

## 환경 설정

### Conda 환경 생성

Python 3.9 환경을 `myenv`라는 이름으로 생성:

```bash
/opt/anaconda3/bin/conda create -n myenv python=3.9 -y
```

## 환경 정보

- 환경 이름: myenv
- Python 버전: 3.9
- 환경 위치: /opt/anaconda3/envs/myenv

```bash
conda create --name myenv python=3.9
# 기존 가상환경을 비활성화 해야한다.
conda deactivate
conda activate myenv
```

## Jupyter Notebook 설정

### 1. Jupyter 설치

Conda 환경에서 Jupyter Notebook 설치:

```bash
# myenv 환경 활성화
conda activate myenv

# Jupyter 설치
conda install jupyter -y

# 또는 pip 사용
pip install jupyter notebook
```

### 2. Jupyter Kernel 등록

현재 conda 환경을 Jupyter 커널로 등록:

```bash
# ipykernel 설치
conda install ipykernel -y

# 현재 환경을 Jupyter 커널로 등록
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
```

### 3. Jupyter Notebook 실행

```bash
# Jupyter Notebook 실행
jupyter notebook

# 또는 특정 포트에서 실행
jupyter notebook --port=8888
```

### 4. Jupyter Lab (선택사항)

더 현대적인 인터페이스를 원하시면 Jupyter Lab 사용:

```bash
# Jupyter Lab 설치
conda install jupyterlab -y

# Jupyter Lab 실행
jupyter lab
```

### 5. 커널 확인 및 관리

```bash
# 설치된 커널 목록 확인
jupyter kernelspec list

# 커널 삭제 (필요한 경우)
jupyter kernelspec uninstall myenv
```

### VSCode에서 Jupyter 사용

VSCode에서 `.ipynb` 파일을 열면 자동으로 Jupyter 환경이 활성화되며, 등록된 커널을 선택할 수 있습니다.

### 6. 샘플 노트북

프로젝트에 샘플 Jupyter Notebook이 포함되어 있습니다:

- [notebooks/sample.ipynb](notebooks/sample.ipynb) - Python 환경 테스트 및 기본 라이브러리 확인
- [notebooks/iris_knn_analysis.ipynb](notebooks/iris_knn_analysis.ipynb) - 아이리스 데이터 KNN 분류 학습
- [notebooks/iris_logistic_regression.ipynb](notebooks/iris_logistic_regression.ipynb) - 아이리스 데이터 로지스틱 회귀분석

노트북을 실행하려면:

```bash
# notebooks 디렉토리로 이동
cd notebooks

# Jupyter Notebook 실행
jupyter notebook sample.ipynb

# 또는 Jupyter Lab 사용
jupyter lab sample.ipynb
```

## 프로젝트

### 아이리스 데이터 분석 (KNN 모델)

아이리스 데이터셋은 품종을 분류하는 지도 학습(Supervised Learning) 문제의 대표적인 예제입니다.

**학습 내용:**
- K-최근접 이웃(K-Nearest Neighbors, KNN) 알고리즘 이해
- 데이터 탐색 및 시각화 (Pairplot, 박스플롯, 상관관계)
- 데이터 전처리 (train/test split, 표준화)
- 모델 학습 및 평가 (정확도, 분류 리포트, 혼동 행렬)
- 최적의 k 값 찾기
- 새로운 데이터 예측

**노트북:** [notebooks/iris_knn_analysis.ipynb](notebooks/iris_knn_analysis.ipynb)

### 아이리스 데이터 분석 (로지스틱 회귀)

로지스틱 회귀를 사용한 아이리스 품종 분류 예제입니다.

**학습 내용:**
- 로지스틱 회귀(Logistic Regression) 알고리즘 이해
- 다중 클래스 분류 (Multinomial Logistic Regression)
- 모델 계수(가중치) 분석 및 해석
- 교차 검증(Cross-Validation)
- 예측 확률 분석
- ROC Curve 및 AUC 평가
- 정규화 파라미터(C) 최적화
- KNN과의 성능 비교

**노트북:** [notebooks/iris_logistic_regression.ipynb](notebooks/iris_logistic_regression.ipynb)

**필요한 라이브러리:**

```bash
conda activate myenv
conda install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow -y
conda install google-genai
conda install -c conda-forge google-genai 

pip install konlpy
```

## TensorFlow 버전 확인

TensorFlow 설치 및 버전 확인:

```python
import tensorflow as tf
print(f"TensorFlow 버전: {tf.__version__}")
```
