import tensorflow as tf
from tensorflow.keras.layers import UpSampling1D, Flatten
from upinterp1d import *
import numpy as np
from matplotlib import pyplot as plt

RATE = 8192
#a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
a = np.linspace(0, 30*np.pi, RATE)

print(a.shape)

a = np.reshape(a, [-1, RATE, 1])


a = tf.convert_to_tensor(a)


b1 = upinterp1d(2, interpolation='linear')(a)
b2 = upinterp1d(2, interpolation='cubic')(a)
b3 = upinterp1d(2, interpolation='gaussian')(a)
b4 = upinterp1d(2, interpolation='lanczos3')(a)
b5 = upinterp1d(2, interpolation='lanczos5')(a)
b6 = upinterp1d(2, interpolation='nearest')(a)
b7 = upinterp1d(2, interpolation='mitchellcubic')(a)


print(b1)

plt.figure()

b1_ = np.reshape(b1, [-1])
b2_ = np.reshape(b2, [-1])
b3_ = np.reshape(b3, [-1])
b4_ = np.reshape(b4, [-1])
b5_ = np.reshape(b5, [-1])
b6_ = np.reshape(b6, [-1])
b7_ = np.reshape(b7, [-1])

plt.plot(b1_)
plt.plot(b2_)
plt.plot(b3_)
plt.plot(b4_)
plt.plot(b5_)
plt.plot(b6_)
plt.plot(b7_)

"""
for i in range(0,6):
    
    b_ = np.reshape(b['%'] %(i), [-1])

    plt.plot(b_)
"""


