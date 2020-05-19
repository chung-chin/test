import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg') # Avoid error when running in backgroud

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D
from tensorflow.keras.layers import BatchNormalization, Reshape, Add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from conv1dTP import conv1dTP
import h5py

#from upinterp1d import upinterp1d
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_item(f_):
    len_ = int(len(f_['waveform']))
    print(len_)
    m1 = []
    m2 = []
    hp = []
    for i in range(len_):
        key = 'waveform/%d' %i
        k2 = 'waveform/%d/hp' %i

        m1.append(f_[key].attrs['m'][0])
        m2.append(f_[key].attrs['m'][1])
        hp.append(f_[k2][:])

    return m1, m2, hp

def deidblock(X, ks, filters):

    Xsc = X              # Keep the input value for skipping connection
    F1, F2, F3 = filters # Retrieve Filters

    # 1st component
    X = conv1dTP(filters = F1, kernel_size=ks, strides=1,
    activation='tanh', padding='same', use_bias=False)(X)
    X = BatchNormalization()(X)

    # 2nd component
    X = conv1dTP(filters = F2, kernel_size=ks, strides=1,
    activation='tanh', padding='same', use_bias=False)(X)
    X = BatchNormalization()(X)

    # 3rd component
    X = conv1dTP(filters = F3, kernel_size=ks, strides=1,
    activation='tanh', padding='same', use_bias=False)(X)
    X = BatchNormalization()(X)

    return Add()([X, Xsc])

NTYPE = 10
RATE = 8192
A = 1
N = 30
batch_size=200
epochs = 300

H5_FILE = '../../../gw_data/bbh_8192_dm0.h5'
f = h5py.File(H5_FILE, 'r')

m1, m2, hp = get_item(f)
m1 = np.array(m1)
m2 = np.array(m2)
hp = np.array(hp)
##gwtrain = pd.read_csv('GW_train_full.csv', index_col=0)
##gw_test = pd.read_csv('GW_test_full.csv' , index_col=0) 

gwtrain = np.array(hp[0:8000])
gw_test = np.array(hp[8001:10000])

xtrain = gwtrain.reshape((-1, RATE, 1))
xtest1 = gw_test.reshape((-1, RATE, 1))

trd0 = xtrain.shape[0]
ted0 = xtest1.shape[0]
print(trd0,ted0)

trbig = np.max(xtrain,axis=1)
tebig = np.max(xtest1,axis=1)

for i in range(trd0): xtrain[i,:,0] = xtrain[i,:,0]/trbig[i,0]
for i in range(ted0): xtest1[i,:,0] = xtest1[i,:,0]/tebig[i,0]

##tot = 1200
##trt = tot - trd0

##xtrain = hp[:trt]
##xtest1 = hp[trt:]

##m1_train = m1[:trt]
##m1_test = m1[trt:]

##m2_train = m2[:trt]
##m2_test = m2[trt:]
##xtrain = np.concatenate((xtrain, xtraid), axis=0)

PLOT = True
if PLOT:
    n=135
    #m1_ = m1_train[n]
    #m2_ = m2_train[n]
    plt.figure()
    plt.plot(xtrain[n])
    plt.savefig('./xtrain_%s.pdf' %str(n))



enksize = 10
inksize = 3
enfilters = [64,32,32,16,16,8,8]
endilates = [1,2,4,8,16,32]
deksize = 10
defilters = [8,16,16,32,32,64,1]
midense = 512
latent_dim = 32

### encoder
inputs = Input(shape=(RATE, 1))
x = inputs

for filters in enfilters:
#   for drate in endilates:
#       x = Conv1D(filters=filters, kernel_size=inksize, padding="same",
#                  activation="tanh", dilation_rate=drate, use_bias=False)(x)
    
    x = Conv1D(filters=filters, kernel_size=enksize, strides=2,
               activation='tanh', padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    #x = LayerNormalization()(x)

x_shape = K.int_shape(x)

x = Flatten()(x)
#x = Dense(midense)(x)

xoutputs = Dense(latent_dim)(x)

xencoder = Model(inputs, xoutputs)
xencoder.summary()
#plot_model(xencoder, to_file='xencoder.png', show_shapes=True)

### decoder
xlatent = Input(shape=latent_dim)

#x = Dense(midense)(xlatent)
x = Dense(x_shape[1] * x_shape[2], use_bias=False)(xlatent)
x = Reshape((x_shape[1], x_shape[2]))(x)

for filters in defilters:
    x = BatchNormalization()(x)
    #x = LayerNormalization()(x)
    x = conv1dTP(filters=filters, kernel_size=deksize, strides=2,
    activation='tanh', padding='same', use_bias=False)(x)
#   x = upinterp1d(interpolation='gaussian')(x)
    x = deidblock(x,3,[filters,filters,filters])

#x = BatchNormalization()(x)

xdecoder = Model(xlatent, x)
xdecoder.summary()
#plot_model(xdecoder, to_file='xdecoder.png', show_shapes=True)

### autoencoder
xautoencoder = Model(inputs, xdecoder(xencoder(inputs)))
xautoencoder.summary()
#plot_model(xautoencoder, to_file='xautoencoder.png', show_shapes=True)

xautoencoder.compile(loss='mse', optimizer='adam')
hist = xautoencoder.fit(xtrain, xtrain, batch_size=batch_size, epochs=epochs,
                        shuffle=True, validation_data=(xtest1, xtest1))

ypred = xautoencoder.predict(xtest1)

plt.figure()
plt.plot(xtest1[1,:,0], color='blue')
plt.plot(ypred[ 1,:,0], color='red' )
plt.title("epoch: %s" %(epochs))
#plt.show()
plt.savefig('gwmatch1.pdf')

plt.figure()
plt.plot(xtest1[100,:,0], color='blue')
plt.plot(ypred[ 100,:,0], color='red' )
plt.title("epoch: %s" %(epochs))
#plt.show()
plt.savefig('gwmatch2.pdf')

# list all data in history
print(hist.history.keys())

# summarize history for loss
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
#plt.show()
plt.savefig('loss.pdf')
