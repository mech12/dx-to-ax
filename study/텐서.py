# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# 1. TensorFlow 버전 출력
print("=" * 60)
print("1. TensorFlow 버전 정보")
print("=" * 60)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"Keras 버전: {tf.keras.__version__}")
print(f"GPU 사용 가능 여부: {tf.config.list_physical_devices('GPU')}")
print()

# 2. 기본 텐서 생성
print("=" * 60)
print("2. 기본 텐서 생성")
print("=" * 60)

# 2-1. 스칼라 (0차원 텐서)
scalar = tf.constant(42)
print(f"스칼라: {scalar}")
print(f"  shape: {scalar.shape}, dtype: {scalar.dtype}, ndim: {scalar.ndim}")
print()

# 2-2. 벡터 (1차원 텐서)
vector = tf.constant([1, 2, 3, 4, 5])
print(f"벡터: {vector}")
print(f"  shape: {vector.shape}, dtype: {vector.dtype}, ndim: {vector.ndim}")
print()

# 2-3. 행렬 (2차원 텐서)
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])
print(f"행렬:\n{matrix}")
print(f"  shape: {matrix.shape}, dtype: {matrix.dtype}, ndim: {matrix.ndim}")
print()

# 2-4. 3차원 텐서
tensor_3d = tf.constant([[[1, 2],
                          [3, 4]],
                         [[5, 6],
                          [7, 8]]])
print(f"3차원 텐서:\n{tensor_3d}")
print(f"  shape: {tensor_3d.shape}, dtype: {tensor_3d.dtype}, ndim: {tensor_3d.ndim}")
print()

# 3. 특수 텐서 생성
print("=" * 60)
print("3. 특수 텐서 생성")
print("=" * 60)

# 3-1. 영(0) 텐서
zeros = tf.zeros([2, 3])
print(f"영 텐서 (2x3):\n{zeros}")
print()

# 3-2. 일(1) 텐서
ones = tf.ones([3, 2])
print(f"일 텐서 (3x2):\n{ones}")
print()

# 3-3. 단위 행렬
identity = tf.eye(3)
print(f"단위 행렬 (3x3):\n{identity}")
print()

# 3-4. 난수 텐서 (정규분포)
random_normal = tf.random.normal([2, 3], mean=0.0, stddev=1.0)
print(f"정규분포 난수 텐서 (2x3):\n{random_normal}")
print()

# 3-5. 난수 텐서 (균등분포)
random_uniform = tf.random.uniform([2, 3], minval=0, maxval=10)
print(f"균등분포 난수 텐서 (2x3):\n{random_uniform}")
print()

# 4. 텐서 연산
print("=" * 60)
print("4. 텐서 연산")
print("=" * 60)

a = tf.constant([[1.0, 2.0],
                 [3.0, 4.0]])
b = tf.constant([[5.0, 6.0],
                 [7.0, 8.0]])

# 4-1. 덧셈
add_result = tf.add(a, b)  # 또는 a + b
print(f"덧셈 (a + b):\n{add_result}\n")

# 4-2. 뺄셈
sub_result = tf.subtract(a, b)  # 또는 a - b
print(f"뺄셈 (a - b):\n{sub_result}\n")

# 4-3. 곱셈 (원소별)
mul_result = tf.multiply(a, b)  # 또는 a * b
print(f"원소별 곱셈 (a * b):\n{mul_result}\n")

# 4-4. 행렬 곱셈
matmul_result = tf.matmul(a, b)  # 또는 a @ b
print(f"행렬 곱셈 (a @ b):\n{matmul_result}\n")

# 4-5. 전치
transpose_result = tf.transpose(a)
print(f"전치 (a^T):\n{transpose_result}\n")

# 5. 텐서 변환
print("=" * 60)
print("5. 텐서 변환")
print("=" * 60)

# 5-1. NumPy 배열로 변환
numpy_array = a.numpy()
print(f"NumPy 배열로 변환:\n{numpy_array}")
print(f"  타입: {type(numpy_array)}\n")

# 5-2. NumPy 배열에서 텐서로 변환
np_data = np.array([[1, 2, 3],
                    [4, 5, 6]])
tensor_from_numpy = tf.constant(np_data)
print(f"NumPy → 텐서 변환:\n{tensor_from_numpy}\n")

# 5-3. 형태 변경 (Reshape)
original = tf.constant([1, 2, 3, 4, 5, 6])
reshaped = tf.reshape(original, [2, 3])
print(f"원본 텐서: {original}")
print(f"재구성 (2x3):\n{reshaped}\n")

# 6. 텐서 인덱싱 및 슬라이싱
print("=" * 60)
print("6. 텐서 인덱싱 및 슬라이싱")
print("=" * 60)

tensor = tf.constant([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
print(f"원본 텐서:\n{tensor}\n")

# 6-1. 특정 원소 접근
element = tensor[1, 2]
print(f"tensor[1, 2] = {element}\n")

# 6-2. 행 선택
row = tensor[0, :]
print(f"첫 번째 행: {row}\n")

# 6-3. 열 선택
col = tensor[:, 1]
print(f"두 번째 열: {col}\n")

# 6-4. 슬라이싱
slice_result = tensor[0:2, 1:3]
print(f"슬라이싱 [0:2, 1:3]:\n{slice_result}\n")

# 7. 집계 함수
print("=" * 60)
print("7. 집계 함수")
print("=" * 60)

data = tf.constant([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])
print(f"데이터:\n{data}\n")

print(f"합계: {tf.reduce_sum(data)}")
print(f"평균: {tf.reduce_mean(data)}")
print(f"최댓값: {tf.reduce_max(data)}")
print(f"최솟값: {tf.reduce_min(data)}")
print()

# 축(axis)을 지정한 집계
print(f"행 기준 합계 (axis=0): {tf.reduce_sum(data, axis=0)}")
print(f"열 기준 합계 (axis=1): {tf.reduce_sum(data, axis=1)}")
print()

# 8. 변수 (Variable)
print("=" * 60)
print("8. 변수 (tf.Variable)")
print("=" * 60)

# 변수는 학습 가능한 파라미터를 저장하는 데 사용됨
variable = tf.Variable([[1.0, 2.0],
                       [3.0, 4.0]])
print(f"변수 초기값:\n{variable}\n")

# 변수 값 변경
variable.assign([[10.0, 20.0],
                [30.0, 40.0]])
print(f"변수 값 변경 후:\n{variable}\n")

# 특정 위치 값 변경
variable[0, 0].assign(100.0)
print(f"variable[0, 0] = 100.0 후:\n{variable}\n")

print("=" * 60)
print("TensorFlow 기본 텐서 예제 완료!")
print("=" * 60)
