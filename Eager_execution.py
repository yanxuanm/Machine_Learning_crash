from __future__ import absolute_import, division, print_function

import tensorflow as tf 
tf.enable_eager_execution()

tf.executing_eagerly()  # => True 

x = [[2.]]
m = tf.matmul(x, x) 
print("hello, {}".format(m))


a = tf.constant([[1, 2], [3, 4]])
print(a)

b = tf.add(a, 1)
print(b)

print(a*b) 

import numpy as np 
c = np.multiply(a, b)
print(c)

print(a.numpy())

def fizzbuzz(max_num): 
	conter = tf.constant(0) 
	max_num = tf.convert_to_tensor(max_num)
	for num in range(max_num.numpy()): 
		num = tf.constant(num) 
		if int(num % 3) == 0 and int(num % 5) == 0:
			print('FizzBuzz') 
		elif int(num % 3 ) == 0:
			print('Fizz') 
		elif int(num % 5) == 0:
			print('Buzz')
		else:
			print(num) 
		counter += 1
	return counter