import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. 데이터 생성 (공부 시간과 시험 점수의 관계)
np.random.seed(42)
study_hours = np.random.uniform(1, 10, 100)  # 1~10시간 사이의 공부 시간
# 점수 = 50 + 5 * 공부시간 + 노이즈
test_scores = 50 + 5 * study_hours + np.random.normal(0, 5, 100)

# 데이터프레임 생성
df = pd.DataFrame({
    '공부시간': study_hours,
    '시험점수': test_scores
})

print("--- 1. 원본 데이터 (처음 10개) ---")
print(df.head(10))
print(f"\n데이터 개수: {len(df)}")

# 2. 데이터 분할 (학습용 70%, 테스트용 30%)
X = df[['공부시간']].values  # 독립변수 (2차원 배열로 변환)
y = df['시험점수'].values     # 종속변수

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n--- 2. 데이터 분할 결과 ---")
print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# 3. 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- 3. 모델 학습 완료 ---")
print(f"회귀 계수 (기울기): {model.coef_[0]:.2f}")
print(f"절편: {model.intercept_:.2f}")
print(f"회귀식: 시험점수 = {model.intercept_:.2f} + {model.coef_[0]:.2f} × 공부시간")

# 4. 예측
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 5. 모델 성능 평가
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n--- 4. 모델 성능 평가 ---")
print(f"[학습 데이터]")
print(f"  MSE (평균제곱오차): {train_mse:.2f}")
print(f"  RMSE (평균제곱근오차): {train_rmse:.2f}")
print(f"  MAE (평균절대오차): {train_mae:.2f}")
print(f"  R² (결정계수): {train_r2:.4f}")

print(f"\n[테스트 데이터]")
print(f"  MSE (평균제곱오차): {test_mse:.2f}")
print(f"  RMSE (평균제곱근오차): {test_rmse:.2f}")
print(f"  MAE (평균절대오차): {test_mae:.2f}")
print(f"  R² (결정계수): {test_r2:.4f}")

# 6. 새로운 데이터로 예측
new_study_hours = np.array([[3], [5], [7], [9]])
predictions = model.predict(new_study_hours)

print("\n--- 5. 새로운 데이터 예측 ---")
for hours, score in zip(new_study_hours.flatten(), predictions):
    print(f"공부시간 {hours}시간 → 예상 점수: {score:.1f}점")

# 7. 시각화
plt.figure(figsize=(12, 5))

# 7-1. 학습 데이터 시각화
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.5, label='학습 데이터')
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='회귀선')
plt.xlabel('공부시간 (시간)')
plt.ylabel('시험점수 (점)')
plt.title(f'학습 데이터 - 선형 회귀\nR² = {train_r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

# 7-2. 테스트 데이터 시각화
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.5, color='green', label='테스트 데이터')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='회귀선')
plt.xlabel('공부시간 (시간)')
plt.ylabel('시험점수 (점)')
plt.title(f'테스트 데이터 - 선형 회귀\nR² = {test_r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=150, bbox_inches='tight')
print("\n그래프가 'linear_regression_result.png' 파일로 저장되었습니다.")
plt.show()