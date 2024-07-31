import numpy as np
import tensorflow as tf
"""x = tf.Variable([1,2,3,4],dtype=tf.float32)
y = tf.Variable([5,6,7,8],dtype=tf.float32)
u = tf.gradients(y,x)
print(u)"""

"""
X = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]]).T
Y = np.array([[21,22,23,24,25,26,27,28,29,30],[31,32,33,34,35,36,37,38,39,40]]).T
Z = np.concatenate([X,Y],axis=1)
print(Z)
lb = Z.min(axis=0)
print(lb)
"""
"""X = np.array([[1,2,3,4,5]])
Y = np.array([[2,3,4]])
Z = np.tile(X[0:1,:],(3,1))
print(Z)"""
"""X = np.array([[1],
              [2],
              [3],
              [4],
              [5]])
Y = np.array([[2,3,4]])
Z = np.tile(X,(3,1))
print(Z)"""
"""N = 15
T = 4
N_train = 15
z = np.random.choice(N*T, N_train, replace = False)
print(z)"""
"""[50 40 11 43 14 10 53 47 29 17  3 54 39 57 35]
"""
"""from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())"""
print(tf.__version__)