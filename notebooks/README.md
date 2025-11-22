# Jupyter Notebooks

이 디렉토리에는 아이리스 데이터 분석을 위한 다양한 머신러닝 알고리즘 학습 노트북이 포함되어 있습니다.

## 노트북 목록

### 1. sample.ipynb
Python 환경 테스트 및 기본 라이브러리 확인용 샘플 노트북

### 2. iris_knn_analysis.ipynb
K-최근접 이웃(KNN) 알고리즘을 사용한 아이리스 품종 분류

**주요 내용:**
- KNN 알고리즘 원리 이해
- 최적의 k 값 찾기
- 거리 기반 분류

### 3. iris_logistic_regression.ipynb
로지스틱 회귀 알고리즘을 사용한 아이리스 품종 분류

**주요 내용:**
- 로지스틱 회귀 원리 이해
- 모델 계수 분석 및 해석
- 정규화 파라미터 최적화
- ROC Curve 및 AUC 평가

## 환경 설정

```bash
# 환경 활성화
conda activate myenv

# 필요한 라이브러리 설치
conda install numpy pandas matplotlib seaborn scikit-learn -y

# Jupyter 실행
jupyter notebook
```

## KNN vs 로지스틱 회귀 비교

### 알고리즘 작동 원리

#### KNN (K-Nearest Neighbors)
- **비모수적 방법**: 데이터의 분포를 가정하지 않음
- **인스턴스 기반 학습**: 새로운 데이터를 분류할 때 가장 가까운 k개의 이웃을 찾아 다수결로 결정
- **거리 측정**: 유클리드 거리, 맨하탄 거리 등을 사용
- **학습 단계 없음** (Lazy Learning): 훈련 데이터를 그대로 저장하고 예측 시에만 계산

#### 로지스틱 회귀 (Logistic Regression)
- **모수적 방법**: 선형 관계를 가정
- **확률 기반 모델**: 선형 결합 후 시그모이드/소프트맥스 함수로 확률 계산
- **가중치 학습**: 각 특성의 영향력(계수)을 학습
- **학습 단계 필요** (Eager Learning): 훈련 데이터로 모델 파라미터를 학습

### 주요 차이점 비교표

| 비교 항목 | KNN | 로지스틱 회귀 |
|----------|-----|--------------|
| **학습 속도** | ⚡⚡ 매우 빠름 (학습 없음) | ⚡ 빠름 |
| **예측 속도** | 🐌 느림 (모든 데이터와 거리 계산) | ⚡⚡ 매우 빠름 |
| **메모리 사용** | ❌ 많음 (모든 훈련 데이터 저장) | ✅ 적음 (계수만 저장) |
| **해석 가능성** | ❌ 낮음 | ✅ 높음 (계수로 영향력 파악) |
| **확률 출력** | ⚠️ 단순함 (k개 중 비율) | ✅ 정교함 (확률 모델) |
| **결정 경계** | 비선형 (복잡한 패턴 학습 가능) | 선형 (단순한 경계) |
| **특성 스케일링** | ✅ 필수 (거리 계산에 영향) | ✅ 권장 (수렴 속도 향상) |
| **하이퍼파라미터** | k (이웃 개수) | C (정규화), solver, penalty |
| **과적합 위험** | k가 작을 때 높음 | C가 클 때 높음 |
| **큰 데이터셋** | ❌ 비효율적 (메모리, 속도) | ✅ 효율적 |
| **고차원 데이터** | ❌ 차원의 저주 | ⚠️ 특성 선택 필요 |

### 성능 비교 (아이리스 데이터셋)

#### KNN 결과
- **최적 k 값**: 3-7 정도에서 최고 성능
- **테스트 정확도**: 약 96-100%
- **장점**:
  - 비선형 경계 학습 가능
  - 하이퍼파라미터가 단순함
- **단점**:
  - 예측 속도가 느림
  - 왜 그렇게 분류했는지 설명하기 어려움

#### 로지스틱 회귀 결과
- **최적 C 값**: 1-10 정도에서 최고 성능
- **테스트 정확도**: 약 96-100%
- **장점**:
  - 예측 속도가 매우 빠름
  - 각 특성의 중요도를 계수로 확인 가능
  - 확률 출력으로 불확실성 측정
- **단점**:
  - 선형 경계만 학습 (복잡한 패턴에는 특성 공학 필요)

### 언제 무엇을 사용할까?

#### KNN을 선택하는 경우
✅ 데이터셋이 작거나 중간 크기일 때
✅ 복잡한 비선형 패턴이 있을 때
✅ 예측 속도가 중요하지 않을 때
✅ 빠른 프로토타이핑이 필요할 때

#### 로지스틱 회귀를 선택하는 경우
✅ 데이터셋이 클 때
✅ 빠른 예측이 필요할 때
✅ 모델 해석이 중요할 때
✅ 확률 출력이 필요할 때
✅ 프로덕션 환경에 배포할 때
✅ 베이스라인 모델로 사용할 때

### 실무 활용 팁

#### KNN 활용 시
```python
# 최적의 k 값 찾기
for k in range(1, 31):
    model = KNeighborsClassifier(n_neighbors=k)
    # 교차 검증으로 평가

# 특성 스케일링 필수
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 로지스틱 회귀 활용 시
```python
# 정규화 강도 조절
model = LogisticRegression(C=1.0)

# 계수 확인으로 특성 중요도 파악
coefficients = model.coef_

# 확률 출력 활용
probabilities = model.predict_proba(X_test)
```

### 앙상블 기법
두 모델을 함께 사용하여 성능을 향상시킬 수도 있습니다:

```python
from sklearn.ensemble import VotingClassifier

# 투표 분류기
ensemble = VotingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('lr', LogisticRegression())
    ],
    voting='soft'  # 확률 기반 투표
)
```

## 다음 단계

학습을 마친 후 다음 알고리즘들을 시도해보세요:

1. **Decision Tree** - 의사결정 트리 (해석이 매우 쉬움)
2. **Random Forest** - 랜덤 포레스트 (앙상블, 높은 성능)
3. **SVM** - 서포트 벡터 머신 (마진 최대화)
4. **Naive Bayes** - 나이브 베이즈 (확률 기반, 빠름)
5. **Neural Network** - 신경망 (복잡한 패턴 학습)

## 참고 자료

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [KNN 공식 문서](https://scikit-learn.org/stable/modules/neighbors.html)
- [Logistic Regression 공식 문서](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
