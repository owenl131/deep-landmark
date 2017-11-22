
# coding: utf-8

# ### Deep Convolutional Network Cascade for Facial Point Detection
# with significant modifications for simplification

# In[1]:


from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Reshape, Permute, Lambda
from keras.models import Model
import numpy as np


# In[2]:


#%matplotlib inline
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# In[ ]:


# 0 for F1
# 1 for EN1
# 2 for NM1
# 3 for all
to_train = 1


# In[4]:


def show_landmark(face, landmark):
    face_copied = face.copy().astype(np.uint8)
    for l in landmark:
        x = l[0]
        y = l[1]
        xx = int(face.shape[0]*x)
        yy = int(face.shape[1]*y)
        cv2.circle(face_copied, (xx, yy), 2, (255,255,255), -1)
    return face_copied


# In[5]:


def break_to_units(lyr, units_y, units_x):
    # break a 2D array into NxN units
    shape = (None, int(lyr.shape[1]), int(lyr.shape[2]), int(lyr.shape[3]))
    lyr = Reshape(
        (units_y, shape[1]//units_y, shape[2], shape[3])
    )(lyr)
    lyr = Permute((1, 3, 2, 4))(lyr)
    lyr = Reshape(
        (units_x*units_y, shape[2]//units_x, shape[1]//units_y, shape[3])
    )(lyr)
    lyr = Permute((1, 3, 2, 4))(lyr)
    return lyr

def recombine_units(lyr, units_y, units_x):
    shape = (None, None, 
             int(lyr.shape[2]), int(lyr.shape[3]), int(lyr.shape[4]))
    lyr = Permute((2, 1, 3, 4))(lyr)
    lyr = Reshape((shape[2], units_y, shape[3]*units_x, shape[4]))(lyr)
    lyr = Permute((2, 1, 3, 4))(lyr)
    lyr = Reshape((shape[2]*units_y, shape[3]*units_x, shape[4]))(lyr)
    return lyr

def abs_layer(lyr):
    lyr = Lambda(lambda x: K.abs(x))(lyr)
    return lyr


# In[6]:


def get_C(lyr, kernel_len, number, p, q):
    lyr = break_to_units(lyr, p, q)
    lyr = Convolution3D(
        filters=number, 
        kernel_size=(p*q,kernel_len,kernel_len),
        strides=(1,1,1),
        padding='same'
    )(lyr)
    lyr = recombine_units(lyr, p, q)
    lyr = Activation('tanh')(lyr)
    return lyr

def get_CR(lyr, kernel_len, number, p, q):
    lyr = get_C(lyr, kernel_len, number, p, q)
    lyr = abs_layer(lyr)
    return lyr

def get_MP(lyr, side_len, p, q):
    lyr = MaxPooling2D(
        pool_size=(side_len, side_len)
    )(lyr)
    #print('mp', lyr.shape)
    #lyr = break_to_units(lyr, p, q)
    #print('mp', lyr.shape)
    #lyr = Convolution3D(
    #    filters=1,
    #    kernel_size=(1,1,1),
    #    strides=(p*q,1,1)
    #)(lyr)
    #print('mp', lyr.shape)
    #lyr = recombine_units(lyr, p, q)
    return lyr

def get_FC(lyr, size):
    lyr = Dense(size)(lyr)
    lyr = Activation('tanh')(lyr)
    return lyr


# In[9]:


def get_s0():
    inp = Input((48, 48, 1))
    lyr = get_CR(inp, 4, 20, 2, 2)
    lyr = get_MP(lyr, 2, 2, 2)
    lyr = get_CR(lyr, 3, 20, 2, 2)
    lyr = get_MP(lyr, 2, 2, 2)
    lyr = get_CR(lyr, 3, 60, 3, 3)
    lyr = get_MP(lyr, 2, 3, 3)
    lyr = get_CR(lyr, 2, 80, 2, 2)
    lyr = Flatten()(lyr)
    lyr = get_FC(lyr, 120)
    lyr = get_FC(lyr, 10)
    model = Model(inp, lyr)
    return model

def get_s1():
    inp = Input((40, 48, 1))
    lyr = get_CR(inp, 4, 20, 1, 1)
    lyr = get_MP(lyr, 2, 1, 1)
    lyr = get_CR(lyr, 3, 20, 2, 2)
    lyr = get_MP(lyr, 2, 2, 2)
    lyr = get_CR(lyr, 3, 60, 2, 3)
    lyr = get_MP(lyr, 2, 2, 3)
    lyr = get_CR(lyr, 2, 80, 1, 2)
    lyr = Flatten()(lyr)
    lyr = get_FC(lyr, 100)
    lyr = get_FC(lyr, 6)
    model = Model(inp, lyr)
    return model

# to check that models can be created
get_s0()
get_s1()


# In[10]:


train_set_info = 'dataset/trainImageList.txt'
trainfile = open(train_set_info, 'r')
x = []
y = []

for line in trainfile:
    info = line.rstrip().split(' ')
    im = cv2.imread('dataset/' + info[0].replace('\\', '/'))
    xy = np.array(info[1:5], dtype=np.int)
    trimmed = im[xy[2]:xy[3], xy[0]:xy[1]]
    trimmed = cv2.resize(trimmed, (48, 48))
    trimmed = cv2.cvtColor(trimmed, cv2.COLOR_BGR2GRAY)
    trimmed = np.reshape(trimmed, trimmed.shape + (1,))
    data = np.array(info[5:], dtype=np.float32)
    data = np.reshape(data, (5, 2))
    for i in range(data.shape[0]):
        data[i][0] = (data[i][0] - xy[0])/(xy[1] - xy[0])
        data[i][1] = (data[i][1] - xy[2])/(xy[3] - xy[2])
    data = np.reshape(data, (10,))
    x.append(trimmed)
    y.append(data)



# In[ ]:


if to_train == 0 or to_train == 3:
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    model = get_s0()
    model.summary()
    model.compile('sgd', 'mse')
    model.fit(x, y, epochs=10)
    model.save_weights('saved_f1.h5')


# In[ ]:


if to_train == 1 or to_train == 3:
    for i in range(len(x)):
        x[i] = x[i][:40, :]
        y[i] = y[i][:6]
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    model = get_s1()
    model.summary()
    model.compile('sgd', 'mse')
    model.fit(x, y, epochs=10)
    model.save_weights('saved_en1.h5')


# In[ ]:


if to_train == 2 or to_train == 3:
    for i in range(len(x)):
        x[i] = x[i][-40:, :]
        y[i] = y[i][-6:]
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    model = get_s1()
    model.summary()
    model.compile('sgd', 'mse')
    model.fit(x, y, epochs=10)
    model.save_weights('saved_nm1.h5')

