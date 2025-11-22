# 케라스 - 텐서 설치시 자동설치( 텐서 내부에 편입) , 
# 파이토치( LLM 개발자들이 사용 )  - 별도 설치
import tensorflow as tf
from tensorflow import keras

# model =>  학습된 데이타 = 신경망 = 네트웍  : 동일한 의미로 쓰인다.
# 심층신경망 :  신경망 레이어를 겹겹이 쌓았다. 
from tensorflow.keras.models import Sequential
# layers.Dense => 겹겹히 쌓으면 학습이 잘 되는데 너무 쌓으면 과대적합이 됨. 
# 적당히 쌓아야 한다. 답은 아무도 모른다.
from tensorflow.keras.layers import Dense, Flatten

import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
# Keras API를 통해 MNIST 데이터셋 로드
# 미국 우편국에서 모은 7만개 손글씨 자료
# 이미지를 numpy 배열로 바꿔야 한다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 훈련셋과 테스트셋을 쪼개서 훈련셋이 과대적합되는것을 막는다.
print("\n" + "="*60)
print("MNIST 데이터셋 정보")
print("="*60)

print("\n[데이터 형태 (Shape)]")
print(f"  x_train (훈련 이미지): {x_train.shape}")
print(f"    → {x_train.shape[0]:,}개 샘플, {x_train.shape[1]}x{x_train.shape[2]} 픽셀")
print(f"  y_train (훈련 레이블): {y_train.shape}")
print(f"    → {y_train.shape[0]:,}개 레이블")
print()
print(f"  x_test (테스트 이미지): {x_test.shape}")
print(f"    → {x_test.shape[0]:,}개 샘플, {x_test.shape[1]}x{x_test.shape[2]} 픽셀")
print(f"  y_test (테스트 레이블): {y_test.shape}")
print(f"    → {y_test.shape[0]:,}개 레이블")

print("\n[전체 요소 개수 (Size)]")
print(f"  x_train 전체 픽셀 수: {x_train.size:,} = {x_train.shape[0]:,} × {x_train.shape[1]} × {x_train.shape[2]}")
print(f"  x_test 전체 픽셀 수: {x_test.size:,} = {x_test.shape[0]:,} × {x_test.shape[1]} × {x_test.shape[2]}")
print(f"  y_train 레이블 개수: {y_train.size:,}")
print(f"  y_test 레이블 개수: {y_test.size:,}")

print("\n[데이터 타입 및 값 범위]")
print(f"  x_train 데이터 타입: {x_train.dtype}")
print(f"  x_train 값 범위: {x_train.min()} ~ {x_train.max()}")
print(f"  y_train 데이터 타입: {y_train.dtype}")
print(f"  y_train 클래스: {np.unique(y_train)} (0~9 숫자)")

print("\n[레이블 분포]")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  숫자 {label}: {count:,}개")

print("="*60 + "\n")

# 이미지 픽셀 값 정규화: 0.0 ~ 1.0 범위로 변환
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# 2. MLP 모델 정의 (tf.keras.Sequential)
# MLP는 입력 데이터를 평탄화(Flatten)하는 것이 필수적입니다. 
model = tf.keras.Sequential([
    # 입력 형태(28x28)를 지정하고 1차원 벡터(784)로 변환
    tf.keras.layers.Flatten(input_shape=(28, 28)), 
    
    # 첫 번째 은닉층: 512개의 노드 (Dense Layer)
    tf.keras.layers.Dense(512, activation='relu'),
    
    # 두 번째 은닉층: 256개의 노드 (Dense Layer)
    tf.keras.layers.Dense(256, activation='relu'),
    
    # 출력층: 10개의 클래스(0-9)를 위한 Softmax 활성화
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3. 모델 컴파일
model.compile(optimizer='adam',
              # 손실 함수: 정수형 레이블을 위한 Sparse Categorical Crossentropy
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 모델 구조 출력
print("--- 모델 요약 ---")
model.summary()
print("------------------")

# 4. 모델 학습 (Training)
# history 객체에 학습 과정의 손실 및 정확도 기록
history = model.fit(
    x_train, 
    y_train, 
    epochs=10, 
    batch_size=128, 
    validation_split=0.1  # 학습 데이터의 10%를 검증에 사용
)

# 5. 모델 평가 (Evaluation)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n🎉 테스트 데이터셋 정확도: {test_acc:.4f}')