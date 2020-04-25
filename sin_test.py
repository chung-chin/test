import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Reshape, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model 
from tensorflow.keras import backend as K

from upinterp1d import *

##import os
##os.environ["CUDA_VISIBLE_DEVICES"] = "1"

RATE = 8192
A = 1
N = 30

kernel_size = 16
lay_filters = [32, 64]
latent_dim = 32

#### build sin wave ###
x = np.linspace(0, 30*np.pi, RATE)
y = np.random.uniform(0, np.pi, N)
xx, yy = np.meshgrid(x, y, sparse=True)
X1 = A*(np.sin(xx-yy))

x_in = X1
y_test = X1[10]







