import tensorflow as tf
from tensorflow.keras.layers import UpSampling1D, Flatten
from upinterp1d import *
import numpy as np
from matplotlib import pyplot as plt

RATE = 4096
A = 1
N = 30

x = np.linspace(0, 10*np.pi, RATE)
y = np.random.uniform(0,np.pi,N)
xx, yy = np.meshgrid(x, y, sparse=True)
a1 = A*(np.sin(xx - yy))

print(a1.shape)

a = np.reshape(a1[0], [-1, RATE, 1])

a = tf.convert_to_tensor(a)

b1 = upinterp1d(2, interpolation='linear')(a)
b2 = upinterp1d(2, interpolation='cubic')(a)
b3 = upinterp1d(2, interpolation='gaussian')(a)
b4 = upinterp1d(2, interpolation='lanczos3')(a)
b5 = upinterp1d(2, interpolation='lanczos5')(a)
b6 = upinterp1d(2, interpolation='nearest')(a)
b7 = upinterp1d(2, interpolation='mitchellcubic')(a)

#print(b1)

plt.figure(figsize=(10,6))

a_ = np.reshape(a, [-1])
b1_ = np.reshape(b1, [-1])
b2_ = np.reshape(b2, [-1])
b3_ = np.reshape(b3, [-1])
b4_ = np.reshape(b4, [-1])
b5_ = np.reshape(b5, [-1])
b6_ = np.reshape(b6, [-1])
b7_ = np.reshape(b7, [-1])

print(a_.shape)
print(a_[0:100])
print(b1_.shape)
print(b1_[0:100])

plt.plot(a_, linewidth='1', markersize=5, label='ori')
plt.plot(b1_, linewidth='1', markersize=5, label='linear')
plt.plot(b2_, linewidth='1', markersize=5, label='cubic')
plt.plot(b3_, linewidth='1', markersize=5, label='gaussian')
plt.plot(b4_, linewidth='1', markersize=5, label='lanczos3')
plt.plot(b5_, linewidth='1', markersize=5, label='lanczos5')
plt.plot(b6_, linewidth='1', markersize=5, label='nearest')
plt.plot(b7_, linewidth='1', markersize=5, label='mitchellcubic')
plt.legend()
plt.savefig('sin_test.pdf')


aa = np.linspace(0, 7, 10)
##aa = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

aa_ = np.reshape(aa, [-1, 10, 1])

aa_ = tf.convert_to_tensor(aa_)

bb1 = upinterp1d(2, interpolation='linear')(aa_)
bb2 = upinterp1d(2, interpolation='cubic')(aa_)
bb3 = upinterp1d(2, interpolation='gaussian')(aa_)
bb4 = upinterp1d(2, interpolation='lanczos3')(aa_)
bb5 = upinterp1d(2, interpolation='lanczos5')(aa_)
bb6 = upinterp1d(2, interpolation='nearest')(aa_)
bb7 = upinterp1d(2, interpolation='mitchellcubic')(aa_)


bb1_ = np.reshape(bb1, [-1])
bb2_ = np.reshape(bb2, [-1])
bb3_ = np.reshape(bb3, [-1])
bb4_ = np.reshape(bb4, [-1])
bb5_ = np.reshape(bb5, [-1])
bb6_ = np.reshape(bb6, [-1])
bb7_ = np.reshape(bb7, [-1])

print(aa)
print(bb1_)
print(bb2_)
print(bb3_)
print(bb4_)
print(bb5_)
print(bb6_)
print(bb7_)

plt.figure(figsize=(10, 6))
plt.plot(aa, 'P-', linewidth='1', markersize=5, label='ori')
plt.plot(bb1_, 'P-', linewidth='1', markersize=5, label='linear')
plt.plot(bb2_, 'P-', linewidth='1', markersize=5, label='cubic')
plt.plot(bb3_, 'P-', linewidth='1', markersize=5, label='gaussian')
plt.plot(bb4_, 'P-', linewidth='1', markersize=5, label='lanczos3')
plt.plot(bb5_, 'P-', linewidth='1', markersize=5, label='lanczos5')
plt.plot(bb6_, 'P-', linewidth='1', markersize=5, label='nearest')
plt.plot(bb7_, 'P-', linewidth='1', markersize=5, label='mitchellcubic')
plt.legend()
plt.savefig('array_test.pdf')




