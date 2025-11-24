import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. 데이터 로드 및 전처리
# Keras API를 통해 MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# CNN 입력을 위한 데이터 형태 변환 (채널 추가)
# (60000, 28, 28) -> (60000, 28, 28, 1) [샘플 수, 높이, 너비, 채널(흑백이므로 1)]
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# 이미지 픽셀 값 정규화: 0.0 ~ 1.0 범위로 변환
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. CNN 모델 정의 (tf.keras.Sequential)
model = Sequential([
    # 첫 번째 컨볼루션 레이어: 특징 추출
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    # 풀링 레이어: 특징 축소
    MaxPooling2D(pool_size=(2, 2)),
    
    # 두 번째 컨볼루션 레이어
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    # 풀링 레이어
    MaxPooling2D(pool_size=(2, 2)),
    
    # 드롭아웃 (과적합 방지)
    Dropout(0.25),
    
    # MLP에 연결하기 위해 1차원으로 평탄화
    Flatten(),
    
    # 밀집층 (Fully Connected Layer): 128개의 노드
    Dense(128, activation='relu'),
    
    # 드롭아웃
    Dropout(0.5),
    
    # 출력층: 10개의 클래스, Softmax 활성화
    Dense(10, activation='softmax')
])

# 3. 모델 컴파일
model.compile(optimizer='adam',
              # 손실 함수: 정수형 레이블을 위한 Sparse Categorical Crossentropy
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 모델 구조 출력
print("--- CNN 모델 요약 ---")
model.summary()
print("-----------------------")

# 4. 모델 학습 (Training)
# 더 깊은 모델이므로 epoch 수를 줄이거나 학습률을 조정할 수 있습니다.
history = model.fit(
    x_train, 
    y_train, 
    epochs=15, 
    batch_size=128, 
    validation_split=0.1  # 검증 데이터 10% 사용
)

# 5. 모델 평가 (Evaluation)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n🎉 테스트 데이터셋 정확도: {test_acc:.4f}')