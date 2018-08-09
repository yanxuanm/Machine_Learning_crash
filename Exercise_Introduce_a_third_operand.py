import tensorflow as tf 
try: 
	tf.contrib.eager.enable_eager_execution()
	print("TF imported with eager execution")
except ValueError: 
	print("TF already imported with eager execution!")

primes = tf.constant([2, 3, 5, 7, 11, 13], dtype = tf.int32)
print("primes:", primes.numpy())

ones = tf.constant(1, dtype = tf.int32)
print("ones:", ones.numpy())

just_beyond_primes = tf.add(primes, ones)
print("just_beyond_primes", just_beyond_primes.numpy())

twos = tf.constant(2, dtype = tf.int32)
primes_doubled = tf.multiply(primes, twos)
print("primes_doubled:", primes_doubled.numpy()) 

some_matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype = tf.int32) 
print(some_matrix) 
print("\nvalue of some_matrix is: \n", some_matrix.numpy())

# A scalar (0-D tensor). 
scalar = tf.zeros([])

# A vector with 3 elements. 
vector = tf.zeros([3]) 

# A matrix with 2 rows and 3 columns. 
matrix = tf.zeros([2, 3]) 

print('scalar has shape', scalar.get_shape(), 'and value:\n', scalar.numpy()) 
print('vector has shape', vector.get_shape(), 'and value:\n', vector.numpy()) 
print('matrix has shape', matrix.get_shape(), 'and value:\n', matrix.numpy())

primes_sqared = tf.pow(primes, twos)
neg_ones = tf.constant(-1, dtype = tf.int32)
just_under_primes_squared = tf.add(primes_sqared, neg_ones)
print("just_under_primes_squared: ", just_under_primes_squared) 

# A 3x4 matrix (2-d tensor). 
x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]], dtype = tf.int32) 

# A 4x2 matrix (2-d tensor). 
y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype = tf.int32) 

# Multiply 'x' by 'y'; result is 3x2 matrix. 
matrix_multiply_result = tf.matmul(x, y) 
print(matrix_multiply_result) 

# Create an 8x2 matrix (2-D tensor). 
matrix = tf.constant(
	[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], 
	dtype = tf.int32) 

reshape_2x8_matrix = tf.reshape(matrix, [2, 8]) 
reshape_4x4_matrix = tf.reshape(matrix, [4, 4]) 

print('Original matrix (8x2):')
print(matrix.numpy()) 

print('Reshaped matrix (2x8):') 
print(reshape_2x8_matrix.numpy()) 

print('Reshaped matrix (4x4):') 
print(reshape_4x4_matrix.numpy())

reshaped_2x2x4_tensor = tf.reshape(matrix, [2, 2, 4]) 
one_dimensional_vector = tf.reshape(matrix, [16])

print('Reshaped 3-D tensor (2x2x4):') 
print(reshaped_2x2x4_tensor.numpy()) 

print('1-D vector:') 
print(one_dimensional_vector.numpy())

a = tf.constant([5, 3, 2, 7, 1, 4]) 
b = tf.constant([4, 6, 3]) 

reshaped_a = tf.reshape(a, [2, 3]) 
reshaped_b = tf.reshape(b, [3, 1]) 

result = tf.matmul(reshaped_a, reshaped_b) 
print(result.numpy()) 

# Variables, Initialization and assignment 
# Create a scalar variable with the initial value 3. 
v = tf.contrib.eager.Variable([3]) 

# Create a vector variable of shape [1, 4], with random initial values, 
# sampled from a normal distribution with mean 1 and standard deviation 0.35. 
w = tf.contrib.eager.Variable(tf.random_normal([1, 4], mean = 1.0, stddev = 0.35)) 

print('v:', v.numpy())
print('w:', w.numpy()) 

v = tf.contrib.eager.Variable([3]) 
print(v.numpy()) 

tf.assign(v, [7]) 
print(v.numpy())

v.assign([5]) 
print(v.numpy()) 

v = tf.contrib.eager.Variable([[1, 2, 3], [4, 5, 6]]) 
print(v.numpy()) 

try: 
	print("Assigning [7, 8, 9] to v" )
	v.assign([7, 8, 9]) 
except ValueError as e: 
	print("Exception:", e) 

# Task: Simulate 10 throws of two dice. Store the results in a 10x3 matrix.

die1 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
die2 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))

dice_sum = tf.add(die1, die2)
resulting_matrix = tf.concat(values=[die1, die2, dice_sum], axis=1)

print(resulting_matrix.numpy())