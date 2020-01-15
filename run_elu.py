from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

from scipy import signal
import scipy
import math
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.gridspec import GridSpec

import time
import os, sys

import random
from data_utils_tf import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

random.seed(1)

sys.path.append('..')


create_initial_data = False
no_train = False  # only create initial data
check = True     # check all variables

batch_size = 256
initial_data = []

model_name = 'OURs'
SNR_list = np.linspace(0, 0.4, num=9*2-1).tolist()[::-1]
SNR_int = 100
print(SNR_list)

############# for initial data ###############
def Normolise(data):
    data_array = np.array(data)
    data_array_shape = data_array.shape[0]
    return pd.DataFrame((data_array -np.mean(data_array, axis=1).reshape(data_array_shape,-1))/np.std(data_array, axis=1).reshape(data_array_shape,-1)
                        ,index = data.index)

def one_hot( vec, vals):
    n = len(vec)
    out = np.zeros((n, vals))
    
    for idx, val in enumerate(vec):
        out[idx][val] = 1

    return out

def data_set(input_x, b_size):

    num_input = int(input_x.shape[0] / b_size)
    #print('num_input= ', num_input)
    try:
        label = input_x.shape[1]
        out = np.zeros([num_input, b_size, label])
        #print('new_shape= ', out.shape)
        for i in range(num_input):
            a = b_size * i
            b = a + b_size
            out[i] = input_x[a:b]
        
    except:
        out = np.zeros([num_input, b_size])
        for i in range(num_input):
            a = b_size * i
            b = a + b_size
            out[i] = input_x[a:b]
            
        out = np.reshape(out, [-1,b_size,1])
            

    #print(out, end='\n\n')
    return out

def mkdir_checkdir(path = "/output"):
    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
        print('MKDIR: ' + path + ' successful!')
    else:
        print(path + " have existed!")


import urllib
url  = 'https://dcc.ligo.org/public/0002/T0900288/003/ZERO_DET_high_P.txt'
raw_data=urllib.request.urlopen(url)
ZERO_DET = np.loadtxt(raw_data)


def noise_psd(noise_sample, lensample, fs,low_pass = 20):
    fs = 8192
    NFFT = fs//8
    NOVL = NFFT/2
    noise_sample = np.array(noise_sample)
    psd_window = np.blackman(NFFT)
    data_freqs = np.fft.fftfreq(lensample) * fs
    power_vec_, freqs_ = mlab.psd(noise_sample, Fs = fs, NFFT = NFFT, window=psd_window
                                  , noverlap=NOVL ,sides='onesided')
    slc = (freqs_>low_pass) #& (freqs_<high_pass)
    return np.interp(np.abs(data_freqs), freqs_[slc], power_vec_[slc])

def noise_psd_zero(ZERO_DET, lensample,fs = 8192,low_pass = 20):
    data_freqs = np.fft.fftfreq(lensample) * fs
    slc = (ZERO_DET[:,0]>=low_pass)# & (ZERO_DET[:,0]<=high_pass)
    asd_zero = np.interp(np.abs(data_freqs)
                         , ZERO_DET[:,0][slc]
                         , ZERO_DET[:,1][slc])
    return asd_zero**2

def SNR_MF(data, noise_sample, signal, GW_train_shape, own_noise=1, fs=8192, low_pass=20):
    lensample = data.shape[1]
    try: # Tukey window preferred, but requires recent scipy version 
        dwindow = scipy.signal.get_window(('tukey',1./8),lensample)
    except: # Blackman window OK if Tukey is not available
        dwindow = scipy.signal.get_window('blackman',lensample)
        print('No tukey windowing, using blackman!')
    # FFT
    data_freqs = np.fft.fftfreq(lensample) * fs
    FFT_data = np.fft.fft(data*dwindow) /fs
    FFT_signal = np.fft.fft(signal*dwindow) /fs

    SNR_mf = np.array([])
    for i in range(GW_train_shape):
        # PSD of noise
        if own_noise == 1: power_vec = noise_psd(noise_sample[i,:], lensample, fs,low_pass = low_pass)
        elif own_noise == 0: power_vec = noise_psd_zero(ZERO_DET, lensample,fs = 8192,low_pass = 20)
        optimal = FFT_data[i,:] * FFT_signal[i,:].conjugate() / power_vec
        optimal_time = 2*np.fft.ifft(optimal) * fs

        # -- Normalize the matched filter output
        df = np.abs(data_freqs[1] - data_freqs[0]) # also df=nsample/fs
        sigmasq = 1*(FFT_signal[i,:] * FFT_signal[i,:].conjugate() / power_vec).sum() * df
        sigma0 = np.sqrt(np.abs(sigmasq))
        SNR_complex = (optimal_time) / (sigma0)
        SNR_mf = np.append(SNR_mf, np.max(np.abs(SNR_complex)))
    return SNR_mf

def pos_gap(samples):
    positions = []
    gaps = []
    for sam in samples.values.tolist():
        position = [index for index, value in enumerate(sam) if (sam[index-1] * sam[index]  < 0) & (index != 0)]
        gaps.append([position[i+1] - j for i,j in enumerate(position) if j != position[-1] ])
        positions.append(position)
    return positions, gaps

def creat_data(GW_train, noise1, SNR):
    # GW_train = Normolise(GW_train)
    noise1array = np.array(noise1)
    GW_train_shape = GW_train.shape[0]
    GW_train_index = GW_train.index
    positions, gaps = pos_gap(Normolise(GW_train))
    max_peak = GW_train.max(axis=1)

    sigma = GW_train.max(axis=1) / float(SNR) / noise1array[:GW_train_shape,:].std(axis=1)
    # data = GW_train + np.multiply(noise1array[:GW_train_shape,:], sigma.reshape((GW_train_shape,-1)) )
    signal = GW_train.div(sigma, axis=0)
    data = signal + noise1array[:GW_train_shape,:]
    SNR_mf = SNR_MF(data=data, noise_sample=noise1array[:GW_train_shape,:], signal=signal
                    ,own_noise=1,GW_train_shape=GW_train_shape
                    , fs=8192, low_pass=20)
    data['SNR_mf0'] = SNR_MF(data=data, noise_sample=noise1array[:GW_train_shape,:], signal=signal
                             ,own_noise=0,GW_train_shape=GW_train_shape
                             , fs=8192, low_pass=20)
    data['SNR_mf'] = SNR_mf

    data['mass'] = GW_train_index
    data['positions'] , data['gaps'] = positions, gaps
    data['max_peak'] = max_peak
    data['sigma'] = sigma

    i = 1
    while (i+1)*GW_train_shape <= noise1array.shape[0]:
        noise1array_p = noise1array[i*GW_train_shape:(i+1)*GW_train_shape,:]

        sigma = GW_train.max(axis=1) / float(SNR) / noise1array_p[:GW_train_shape,:].std(axis=1)
        # data_new = GW_train + np.multiply(noise1array_p[:GW_train_shape,:], sigma.reshape((GW_train_shape,-1)) )
        data_new = GW_train.div(sigma, axis=0) + noise1array_p[:GW_train_shape,:]

        data_new['mass'] = GW_train_index
        data_new['positions'] , data_new['gaps'] = positions, gaps
        data_new['max_peak'] = max_peak
        data_new['sigma'] = sigma
        data = pd.concat([data, data_new ])
        i+=1
        print('Loop! ',i-1 , end='')
#         print('{"metric": "LOOP for SNR=%s", "value": %d}' %(str(SNR), int(i-1)) )
        sys.stdout.write("\r")
    return data

################## create initial data ####################
if create_initial_data:
    
    address_ = ('pycbc_inidata')
    mkdir_checkdir(path = "../%s" %address_)

    data_GW_train = pd.read_csv('~/paper/input/waveform/GW_train_full.csv', index_col=0)
    print('The shape of data_GW_train: ' , data_GW_train.shape)
    data_GW_test = pd.read_csv('~/paper/input/waveform/GW_test_full.csv', index_col=0)
    print('The shape of data_GW_test: ' , data_GW_test.shape)


    noise1 = np.load('../input/noise/pycbc_noise1_10000.npy')
    print('The shape of the noise1: ', noise1.shape)

    noise_train = np.load('../input/noise/pycbc_noise2_10000.npy')
    print('The shape of the noise_train: ', noise_train.shape)

    noise2 = np.load('../input/noise/pycbc_noise3_10000.npy')
    print('The shape of the noise2: ', noise2.shape)

    noise_test = np.load('../input/noise/pycbc_noise4_10000.npy')
    print('The shape of the noise_test: ', noise_test.shape)
    
    sampling_freq = 8192
    colomns = [ str(i) for i in range(sampling_freq)] + ['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0']

    train_dict = {}
    test_dict = {}
    SNR_MF_list = []
    
    train_data = np.array(data_GW_train)
    test_data = np.array(data_GW_test)
    peak_samppoint, peak_time = cal_peak_nd(train_data)
    print(peak_samppoint, peak_time)
    rand_times = 14
    train_, train_shift_list = shuffle_data_np(train_data,peak_samppoint, peak_time, rand_times)
    rand_times = 13
    test_, train_shift_list = shuffle_data_np(test_data,peak_samppoint, peak_time, rand_times)

    for snr in SNR_list:
        print()
        print('Create initial data. SNR = ', snr)

        try:
            print(train_dict['%s' %int(snr*SNR_int)].shape)
            print(test_dict['%s' %int(snr*SNR_int)].shape)
        except:
            data_train = creat_data(pd.DataFrame(train_, columns=colomns[:8192]), noise1.astype('float64'), snr)
            print(data_train.shape)
            print(data_train.SNR_mf.mean())
        
            data_test = creat_data(pd.DataFrame(test_, columns=colomns[:8192]), noise2.astype('float64'), snr)
            print(data_test.shape)
            print(data_test.SNR_mf.mean())

            train_dict['%s' %int(snr*SNR_int)] = pd.concat([data_train, pd.DataFrame(noise_train, columns=colomns[:8192]).iloc[:data_train.shape[0],:]])[colomns]
            test_dict['%s' %int(snr*SNR_int)] = pd.concat([data_test, pd.DataFrame(noise_test, columns=colomns[:8192]).iloc[:data_test.shape[0],:]])[colomns]
            print(train_dict['%s' %int(snr*SNR_int)].shape)
            print(test_dict['%s' %int(snr*SNR_int)].shape)

            SNR_MF_list.append(test_dict['%s' %int(snr*SNR_int)].dropna().SNR_mf.values.mean(axis=0))
            print('SNR_MF_list: ',SNR_MF_list,'\n')


############ initial data #############

        initial_data = [train_dict, test_dict]
        
        f = open('../pycbc_inidata/initial_data_%s.pkl' %int(snr*SNR_int), 'wb')
        pickle.dump(initial_data, f)
        initial_data = []
        f.close()
        
        train_dict = {}
        test_dict = {}
        
    np.save('../pycbc_inidata/SNR_MF_list.npy', np.array(SNR_MF_list))



if no_train:
    os._exit(0)


############# CNN model ################
def net(x_, drop_prob, WIDTH, debug=True):
    feature = tf.reshape(x_, [-1, WIDTH,1])

    args = {"padding":'valid', "activation":None, 
            ##"kernel_initializer":tf.truncated_normal_initializer(),
            "kernel_initializer":tf.random_normal_initializer(stddev=.01,dtype=tf.float64),
            ##"bias_initializer":tf.zeros_initializer()
            "bias_initializer":tf.random_normal_initializer(stddev=.01,dtype=tf.float64)}
    
    def conv1d(x, f, k, d, s, p, ps, act):
        out = tf.layers.conv1d(x, filters=f, kernel_size=k, dilation_rate=d, strides=s, **args)
        out = tf.layers.average_pooling1d(out, pool_size= p, strides= ps, padding='valid')
        return act(out)
    
    def conv1d_(x, f, k, d, s, act):
        out = tf.layers.conv1d(x, filters=f, kernel_size=k, dilation_rate=d, strides=s, **args)
        return act(out)


    h1 = conv1d(feature, f=64, k=16, d=1, s=1, p=4, ps=4, act=tf.nn.relu)
    ##h1 = conv1d_(feature, f=64, k=16, d=1, s=1, act=tf.nn.relu)
    h2 = conv1d(h1     , f=128, k=16,  d=2, s=1, p=4, ps=4, act=tf.nn.relu)
    ##h2 = conv1d_(h1     , f=128, k=16, d=2, s=1, act=tf.nn.relu)
    
    h3 = conv1d(h2     , f=256, k=16, d=2, s=1, p=4, ps=4, act=tf.nn.relu)

    dim_ = h3.get_shape().as_list()
    fcnn_ = dim_[1]*dim_[2]
    h3 = tf.reshape(h3, [-1, fcnn_])

    h4 = tf.layers.dense(h3, 1024, activation=tf.nn.relu)
    h4 = tf.reshape(h4, [-1, 1024, 1])

    ##h4 = conv1d(h3     , f=128, k=16,  d=2, s=1, p=4, ps=4, act=tf.nn.relu)
    ##h3 = conv1d_(h2     , f=256, k=16, d=2, s=1, act=tf.nn.relu)
    h5 = conv1d(h4     , f=128, k=32,  d=2, s=1, p=4, ps=4, act=tf.nn.relu)
    ##h4 = conv1d_(h3     , f=512, k=32, d=2, s=1, act=tf.nn.relu)
    
    dim = h5.get_shape().as_list()
    fcnn = dim[1]*dim[2]
    h5 = tf.reshape(h5, [-1, fcnn])
    
    h6          = tf.layers.dense(h5, 128, activation=tf.nn.relu)
    h7          = tf.layers.dense(h6, 64,  activation=tf.nn.relu)
    ##h6          = tf.nn.dropout(h6, rate= drop_prob)
    yhat_linear = tf.layers.dense(h7,  1, activation=None)
    
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))
        print("h2 shape: %s" % (np.array(h2.shape)))
        ##print("h3 shape: %s" % (np.array(h3.shape)))
        print("h4 shape: %s" % (np.array(h4.shape)))
        print("h5 shape: %s" % (np.array(h5.shape)))
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))
    
    return yhat_linear


######################
### Construct TF graph
######################
tf.reset_default_graph()

DIM = 8192
drop_prob = 0
b_size = 256   # batch size
lr = 0.0003    # learning rate
epochs = 40     # max epoch

X = tf.placeholder(tf.float64, shape=(None, DIM))
Y = tf.placeholder(tf.float64, shape=(None, 1))
drop_prob_ = tf.placeholder(tf.float64)
bs         = tf.placeholder(tf.int64)


train_ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size=10000).batch(bs).repeat()
test_ds  = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size=10000).batch(bs)


## Use one common iterator
iter = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
features, labels = iter.get_next()

# create the initialisation operations
train_init_op = iter.make_initializer(train_ds)
test_init_op = iter.make_initializer(test_ds)

logits = net(features, drop_prob, DIM)

# Compute predictions
with tf.name_scope('eval'):
    predict_prob = tf.sigmoid(logits, name="sigmoid_tensor")
    predict_op   = tf.cast( tf.round(predict_prob), tf.int32 )

with tf.name_scope('loss'):
    ## with reduction compared to tf.nn.softmax_cross_entropy_with_logits_v2 
    loss_op = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels)

with tf.name_scope('adam_optimizer'):
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss_op)
    ##optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss_op)

##correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
##accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))

label = tf.reshape(labels,[-1])

_, accuracy    = tf.metrics.accuracy(labels=labels, predictions=predict_op  )
_, sensitivity = tf.metrics.recall(labels=labels, predictions=predict_op  )

_, fp = tf.metrics.false_positives(labels=labels, predictions=predict_op  )
_, fn = tf.metrics.false_negatives(labels=labels, predictions=predict_op  )
_, tp = tf.metrics.true_positives(labels=labels, predictions=predict_op  )
_, tn = tf.metrics.true_negatives(labels=labels, predictions=predict_op  )

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=100)

#################
### check variable
#################
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope("summaries_%s"% var.name.replace("/", "_").replace(":", "_")):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

### check all variables
if check:
    vars = 0
    for v in tf.global_variables():
        print (v)
        vars += np.prod(v.get_shape().as_list())
    print("Whole size: %.3f MB | Var # : %d" % (8*vars/(1024**2), len(tf.global_variables()) ) )

    vars = 0
    for v in tf.trainable_variables():
        print (v)
        #variable_summaries(v)
        vars += np.prod(v.get_shape().as_list())
    print("Model size: %.3f MB | Var # : %d" % (8*vars/(1024**2), len(tf.trainable_variables()) ) )

    vars = 0
    for v in tf.local_variables():
        print (v)
        vars += np.prod(v.get_shape().as_list())
    print("Local var size: %.3f Bytes | Var # : %d" % (8*vars, len(tf.local_variables()) ) )



################# input initial data #####################
##if 0:
with tf.Session() as sess:
    sess.run(init)

    ##for snr in [10]:
    for snr in SNR_list:
        sess.run(tf.local_variables_initializer())
        ##sess.run(init)
        
        address = 'SNR%s' %(int(snr*SNR_int))
        mkdir_checkdir(path = "./%s" %address)
        
        train_acc = []
        train_sen = []
        epoch = 1      # ini epoch
    
        print('Start Training at SNR= %s !' %(snr), '\n')
    
        f = open('../pycbc_inidata/initial_data_%s.pkl' %int(snr*SNR_int), 'rb')
        train_dict_, test_dict_ = pickle.load(f)
        f.close()
    
        train = train_dict_['%s' %int(snr*SNR_int)]
    
        num_examples= train.shape[0]
        print('num_examples: ', num_examples, end='\n\n')
    
        y_train = np.array(~train.sigma.isnull() +0)
        y_train = np.reshape(y_train, [-1,1])
        X_train = np.array(Normolise(train.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
        print('Label for training:', y_train.shape)
        print('Dataset for training:', X_train.shape, end='\n\n')
    
        if 0:
            for i in range(10):
                feature, label = iter.get_next()
                print (feature.shape, label.shape)


        ckpt_path = './SNR%s/model.ckpt' %(int(snr*SNR_int))

        sess.run(train_init_op, feed_dict={X:X_train, Y:y_train, bs:b_size})
        
        steps = int(num_examples/b_size)
        
        while epoch < epochs:
            print('epoch= ', epoch)
            
            ##if epoch == 5:
            ##    lr = 0.0003

            if epoch > 2:
                lr /= (1+0.01*epoch)
                
            for i in range(steps):
                _, loss, acc, sen = sess.run( [optimizer, loss_op, accuracy, sensitivity] )
                print('loss for SNR=%s : %s' %(str(snr),loss))
                
            epoch += 1
            
            if loss < 1e-6:

                        break

            print('accuracy: %s,  sensitivity: %s' %(acc,sen))
            train_acc.append(acc)
            train_sen.append(sen)
            
        save_path = saver.save(sess, ckpt_path)
        print("Model saved in path: %s" % save_path)
        
        np.save('./SNR%s/train_accuracy.npy' %(int(snr*SNR_int)), train_acc)
        np.save('./SNR%s/train_sensitivity.npy' %(int(snr*SNR_int)), train_sen)



###############
### Testing ###
###############
SNR_list_ = np.linspace(0, 0.4, num=9*2-1).tolist()[-10-1::-1]
print('SNR_list_: ', SNR_list_)
with tf.Session() as sess:
    sess.run(init)
    
    auc_frame = []
    fpr_frame = []
    tpr_frame = []
    acc_frame = []
    sen_frame = []

    ##for snr in [10]:
    for snr in SNR_list:
        address = 'SNR%s' %(int(snr*SNR_int))
    
        load_path = saver.restore(sess, './SNR%s/model.ckpt' %(int(snr*SNR_int)))
        print("Model restored from /SNR%s/model.ckpt" %(int(snr*SNR_int))) 

        test_acc = []
        test_sen = []
        tpr_list = []
        fpr_list = []
        auc_list = []

        for snr_ in SNR_list:
            sess.run(tf.local_variables_initializer())
            
            test_acc_ = []
            test_sen_ = []
            test_pre_ = []
            test_lab = []
            
            f_ = open('../pycbc_inidata/initial_data_%s.pkl' %int(snr_*SNR_int), 'rb')
            train_dict__, test_dict__ = pickle.load(f_)
            f_.close()
            
            test  = test_dict__['%s' %int(snr_*SNR_int)]
            
            num_examples= test.shape[0]
            ##print('num_examples: ', num_examples, end='\n\n')
            
            y_test = np.array(~test.sigma.isnull() +0)
            y_test = np.reshape(y_test, [-1,1])
            X_test = np.array(Normolise(test.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
            print('Label for testing:', y_test.shape)
            print('Dataset for testing:', X_test.shape, end='\n\n')
            
            sess.run(test_init_op, feed_dict={X:X_test, Y:y_test, bs:b_size})
            
            steps = int(num_examples/b_size)
            
            for i in range(steps):
                pre, loss, acc, sen, ttp, ttn, tfp, tfn, tlb = sess.run([predict_prob, loss_op, accuracy, sensitivity, tp, tn, fp, fn, label ])
                
                test_pre_.extend(pre)
                test_acc_.append(acc)
                test_sen_.append(sen)
                test_lab.extend(tlb)
            
            test_pre_ = np.array(test_pre_)
            test_pre_ = np.reshape(test_pre_, [-1])
            print('pre_shape: ', test_pre_.shape)
            #h = len(test_pre_)
            test_lab_ = np.array(test_lab)
            print('label_shape: ', test_lab_.shape)
            fpr, tpr, _ = roc_curve(test_lab_,test_pre_)
            print('tpr: ', tpr.shape)
            auc_ = metrics.auc(fpr, tpr)
            
            auc_list.append(auc_)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

            acc_ = np.mean(test_acc_)
            sen_ = np.mean(test_sen_)
            test_acc.append(acc_)
            test_sen.append(sen_)
            
            print('Test for SNR=%s, loss: %s, acc: %s, sen: %s' %(str(snr_),loss,acc_,sen_),end='\n\n')
            
        acc_frame.append(test_acc)
        sen_frame.append(test_sen)
        tpr_frame.append(tpr_list)
        fpr_frame.append(fpr_list)
        auc_frame.append(auc_list)
        
    mkdir_checkdir(path = "./output")
    np.save('./output/test_accuracy',acc_frame)
    np.save('./output/test_sensitivity',sen_frame)
    np.save('./output/tpr',tpr_frame)
    np.save('./output/fpr',fpr_frame)
    np.save('./output/AUC',auc_frame)
    
    print('finished !')




