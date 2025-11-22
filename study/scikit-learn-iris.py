import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Iris 데이터셋 로드
iris = load_iris()

print(type(iris))
print("=" * 60)
print("1. Iris 데이터셋 기본 정보")
print("=" * 60)
print(f"데이터셋 설명:\n{iris.DESCR[:500]}...\n")
print(f"특성(Feature) 이름: {iris.feature_names}")
print(f"타겟(Target) 이름: {iris.target_names}")
print(f"데이터 shape: {iris.data.shape}")
print(f"타겟 shape: {iris.target.shape}")

# 2. 데이터프레임으로 변환
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("\n" + "=" * 60)
print("2. 데이터프레임 미리보기 (처음 10개)")
print("=" * 60)
print(df.head(10))

# 3. 기본 통계 분석
print("\n" + "=" * 60)
print("3. 기본 통계 정보")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("4. 품종별 데이터 개수")
print("=" * 60)
print(df['species_name'].value_counts())

# 4. 품종별 평균값 분석
print("\n" + "=" * 60)
print("5. 품종별 특성 평균값")
print("=" * 60)
species_mean = df.groupby('species_name')[iris.feature_names].mean()
print(species_mean)

# 5. 데이터 분할 (학습용 80%, 테스트용 20%)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("6. 데이터 분할 결과")
print("=" * 60)
print(f"학습 데이터: {X_train.shape[0]}개")
print(f"테스트 데이터: {X_test.shape[0]}개")

# 6. 로지스틱 회귀 모델 학습
print("\n" + "=" * 60)
print("7. 로지스틱 회귀 모델 학습")
print("=" * 60)

lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train, y_train)

lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)

print(f"학습 정확도: {lr_train_acc:.4f} ({lr_train_acc*100:.2f}%)")
print(f"테스트 정확도: {lr_test_acc:.4f} ({lr_test_acc*100:.2f}%)")

# 7. 의사결정나무 모델 학습
print("\n" + "=" * 60)
print("8. 의사결정나무 모델 학습")
print("=" * 60)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model.fit(X_train, y_train)

dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)

dt_train_acc = accuracy_score(y_train, dt_train_pred)
dt_test_acc = accuracy_score(y_test, dt_test_pred)

print(f"학습 정확도: {dt_train_acc:.4f} ({dt_train_acc*100:.2f}%)")
print(f"테스트 정확도: {dt_test_acc:.4f} ({dt_test_acc*100:.2f}%)")

# 8. 상세 분류 리포트 (로지스틱 회귀)
print("\n" + "=" * 60)
print("9. 로지스틱 회귀 - 상세 분류 리포트 (테스트 데이터)")
print("=" * 60)
print(classification_report(y_test, lr_test_pred, target_names=iris.target_names))

# 9. 혼동 행렬 (Confusion Matrix)
print("\n" + "=" * 60)
print("10. 혼동 행렬 (Confusion Matrix)")
print("=" * 60)
cm = confusion_matrix(y_test, lr_test_pred)
print(cm)

# 10. 특성 중요도 분석 (의사결정나무)
print("\n" + "=" * 60)
print("11. 의사결정나무 - 특성 중요도")
print("=" * 60)
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# 11. 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 11-1. 품종별 꽃받침 길이 vs 너비
ax1 = axes[0, 0]
for species_id, species_name in enumerate(iris.target_names):
    mask = df['species'] == species_id
    ax1.scatter(
        df[mask]['sepal length (cm)'],
        df[mask]['sepal width (cm)'],
        label=species_name,
        alpha=0.6,
        s=50
    )
ax1.set_xlabel('Sepal Length (cm)')
ax1.set_ylabel('Sepal Width (cm)')
ax1.set_title('품종별 꽃받침 크기 분포')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 11-2. 품종별 꽃잎 길이 vs 너비
ax2 = axes[0, 1]
for species_id, species_name in enumerate(iris.target_names):
    mask = df['species'] == species_id
    ax2.scatter(
        df[mask]['petal length (cm)'],
        df[mask]['petal width (cm)'],
        label=species_name,
        alpha=0.6,
        s=50
    )
ax2.set_xlabel('Petal Length (cm)')
ax2.set_ylabel('Petal Width (cm)')
ax2.set_title('품종별 꽃잎 크기 분포')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 11-3. 혼동 행렬 시각화
ax3 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=ax3)
ax3.set_xlabel('예측값')
ax3.set_ylabel('실제값')
ax3.set_title('혼동 행렬 (Confusion Matrix)')

# 11-4. 특성 중요도 시각화
ax4 = axes[1, 1]
ax4.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
ax4.set_xlabel('중요도')
ax4.set_title('의사결정나무 - 특성 중요도')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('iris_analysis_result.png', dpi=150, bbox_inches='tight')
print("\n그래프가 'iris_analysis_result.png' 파일로 저장되었습니다.")
plt.show()

# 12. 새로운 데이터 예측 예제
print("\n" + "=" * 60)
print("12. 새로운 데이터 예측 예제")
print("=" * 60)

new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # setosa 같은 특징
    [6.5, 3.0, 5.2, 2.0],  # virginica 같은 특징
    [5.9, 3.0, 4.2, 1.5]   # versicolor 같은 특징
])

predictions = lr_model.predict(new_samples)
probabilities = lr_model.predict_proba(new_samples)

for i, (sample, pred, prob) in enumerate(zip(new_samples, predictions, probabilities)):
    print(f"\n샘플 {i+1}: {sample}")
    print(f"  예측 품종: {iris.target_names[pred]}")
    print(f"  확률 분포: ", end="")
    for j, species in enumerate(iris.target_names):
        print(f"{species}={prob[j]:.2%}", end=" ")
    print()
