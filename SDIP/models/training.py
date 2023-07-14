#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, concatenate
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
from tensorflow.python.framework import ops
import scipy.io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import interpolation


from tensorflow.keras import layers

def VisualGraphs(RstActual,RstBest,ZTuckerRepr,ConvCurve,Original,ColorBands,path):

    clear_output(wait=True)
    [m,n,L]=Original.shape
    
    #Choose 3 uniform bands from low-Rank Tucker Representation of Z
    bands = np.floor( np.linspace(L/4, 3*L/4, num=3)).astype(int)
    xo = ZTuckerRepr[:,:,[bands[0],bands[1],bands[2]]]
    xo[:,:,0] = xo[:,:,0]/np.max(xo[:,:,0])
    xo[:,:,1] = xo[:,:,1]/np.max(xo[:,:,1])
    xo[:,:,2] = xo[:,:,2]/np.max(xo[:,:,2])

    ErrActual = np.divide(np.power(np.sum(np.power(RstActual-Original,2),axis=2),0.5),np.power(np.sum(np.power(Original,2),axis=2),0.5))

    ErrBest = np.divide(np.power(np.sum(np.power(RstBest-Original,2),axis=2),0.5),np.power(np.sum(np.power(Original,2),axis=2),0.5))

    fig, axs = plt.subplots(2,4,figsize=(16,8))
    fig.subplots_adjust(left=.05, bottom=0.1, right=.9, top=0.9, wspace=0.05)

    RGB = Original[:,:,ColorBands]
    RGB = RGBZoom(RGB)
    axs[0,0].imshow(RGB)
    axs[0,0].set_title('Original')
    axs[0,0].axis('off')

    RGB = RstBest[:,:,ColorBands]
    RGB = RGBZoom(RGB)
    axs[0,1].imshow(RGB)
    axs[0,1].set_title('Best Reconstruction')
    axs[0,1].axis('off')

    im = axs[0, 2].imshow(ErrBest, cmap='hot', vmin=0, vmax=1)
    divider = make_axes_locatable(axs[0,2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[0,2].set_title('Relative Error Map\n Best Reconstruction')
    axs[0,2].axis('off')

    axs[0,3].imshow(xo)
    axs[0,3].set_title('3Bands from Tucker \n Representation of Z')
    axs[0,3].axis('off')


    asp = np.diff(axs[1,0].get_xlim())[0] / np.diff(axs[1,0].get_ylim())[0]
    axs[1,0].set_aspect(asp)
    axs[1,0].axis('off')

    RGB = RstActual[:,:,ColorBands]/np.max(RstBest)
    RGB = RGBZoom(RGB)
    axs[1,1].imshow(RGB)
    axs[1,1].set_title('Actual Reconstruction')
    axs[1,1].axis('off')

    im = axs[1, 2].imshow(ErrActual, cmap='hot', vmin=0, vmax=1)
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axs[1,2].set_title('Relative Error Map\n Actual Reconstruction')
    axs[1,2].axis('off')

    axs[1,3].plot(ConvCurve)
    axs[1,3].set_title('Convergence Curve\n (Best/Actual)PSNR: (%.2f,' %np.max(ConvCurve) +'%.2f)'%ConvCurve[-1])
    asp = np.diff(axs[1, 3].get_xlim())[0] / np.diff(axs[1, 3].get_ylim())[0]
    axs[1,3].set_aspect(asp)
    axs[1,3].yaxis.tick_right()
    
    plt.savefig(path+ '/recons.jpg', bbox_inches='tight')


def addGaussianNoise(y,SNR):
    sigma = np.sum(np.power(y,2))/(np.product(y.shape)*10**(SNR/10))
    w = np.random.normal(0, np.sqrt(sigma),size =y.shape)
    return y+w
    # return y+noise_y

def RGBZoom(X):
    
    for i in range(3):
        Y = interpolation.zoom(X[378-200:443-200,133:198,i],198/65)
        X[12:210,12:210,i] = Y
    
    w = 12
    y = [12, 209]
    x = [12, 209]
    X[y[0]-w:y[0],x[0]-w:x[1]+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]:y[1]+1+w,x[0]-w:x[0],0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    
    w = 4
    y = [378-200, 443-200]
    x = [133, 198]    
    X[y[0]-w:y[0],x[0]-w:x[1]+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]:y[1]+1+w,x[0]-w:x[0],0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
            
            
    return X

def CTZoom(X):
    
    for i in range(3):
        Y = interpolation.zoom(X[378-200:443-200,133:198,i],198/65)
        X[12:210,12:210,i] = Y
    
    w = 12
    y = [12, 209]
    x = [12, 209]
    X[y[0]-w:y[0],x[0]-w:x[1]+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]:y[1]+1+w,x[0]-w:x[0],0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    
    w = 4
    y = [378-200, 443-200]
    x = [133, 198]    
    X[y[0]-w:y[0],x[0]-w:x[1]+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[0]-w:y[0],x[0]-w:x[1]+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]:y[1]+1+w,x[0]-w:x[0],0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]:y[1]+1+w,x[0]-w:x[0],2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,0] = np.ones(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,1] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    X[y[1]+1:y[1]+w+1,x[0]:x[1]+w+1,2] = np.zeros(shape=(w,x[1]-x[0]+w+1))
    
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,0] = np.ones(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,1] = np.zeros(shape=(y[1]-y[0]+w+1,w))
    X[y[0]-w:y[1]+1,x[1]+1:x[1]+w+1,2] = np.zeros(shape=(y[1]-y[0]+w+1,w))
            
            
    return X
   
def Hxfunction(x,largo,ancho,profun,H):
    Aux = tf.reshape(x,(largo,ancho,profun))
    Aux = tf.transpose(Aux,perm=[2,1,0])
    Aux = tf.reshape(Aux,(largo*ancho*profun,1))
    
    #print(Aux.shape)
    
    Aux = tf.sparse.sparse_dense_matmul(H,tf.cast(Aux,dtype=tf.float64))
    Aux = tf.reshape(Aux,(1,1,1,H.shape[0]))
    return Aux



class XoLayer(layers.Layer):
    def __init__(self, largo = 256, ancho = 256, profun = 10, fact = 0.3):
        super(XoLayer, self).__init__()

        self.largo  = largo
        self.ancho  = ancho
        self.profun  = profun

        self.largo_fac = tf.cast(tf.math.round(largo*fact),dtype=tf.int32)
        self.ancho_fac = tf.cast(tf.math.round(ancho*fact),dtype=tf.int32)
        self.profun_fac = tf.cast(tf.math.round(profun*fact),dtype=tf.int32)

          
        self.kernel = self.add_weight(shape=(tf.cast(tf.math.round(profun*fact),dtype=tf.int32),tf.cast(tf.math.round(largo*fact),dtype=tf.int32)*tf.cast(tf.math.round(ancho*fact),dtype=tf.int32)),
                             initializer='glorot_normal',#'glorot_normal',
                             trainable=True)
           
        self.Dx    = self.add_weight(shape=(largo, tf.cast(tf.math.round(largo*fact),dtype=tf.int32)),
                             initializer='uniform', # uniform
                             trainable=True)
        self.Dy    = self.add_weight(shape=(ancho,tf.cast(tf.math.round(ancho*fact),dtype=tf.int32)),
                             initializer='uniform',
                             trainable=True)
        self.Dz    = self.add_weight(shape=(profun,tf.cast(tf.math.round(profun*fact),dtype=tf.int32)),
                            initializer='uniform',
                             trainable=True)


        
    def call(self, inputs):
        
        Aux = tf.transpose(tf.matmul(self.Dz,self.kernel))
        Aux = tf.reshape(Aux,( self.largo_fac,self.ancho_fac*self.profun))

        Aux = tf.matmul(self.Dx,Aux)
        Aux = tf.reshape(Aux,(self.largo,self.ancho_fac,self.profun))
        Aux = tf.transpose(Aux,perm=[1,0,2])
        Aux = tf.reshape(Aux,(self.ancho_fac,self.ancho*self.profun))
        Aux = tf.matmul(self.Dy,Aux)
        Aux = tf.reshape(Aux,(self.ancho,self.largo,self.profun))
        Aux = tf.reshape(tf.transpose(Aux,perm=[1,0,2]),(1,self.ancho,self.largo,self.profun))
        
        return  Aux


class RandomInit(layers.Layer):
    def __init__(self, d=256, M=256):
        super(RandomInit, self).__init__()

        self.d = d
        self.M = M
        self.dense = tf.keras.layers.Dense(M**2,use_bias=True,activation='relu')
    def build(self, input_shape):
        self.z = self.add_weight(shape=(1,self.M,self.M,1),trainable=True,initializer='random_normal')
    def call(self, inputs):
        #z = tf.random.normal(shape=(1,self.M,self.M,1))*0.01
        #z = z/tf.reduce_max(z)
        #out = self.dense(z)
        #out = tf.reshape(out,[1,self.M,self.M,1])
        return self.z

    
class RandomNormalInitLayer(tf.keras.layers.Layer):
    def __init__(self, num_params):
        super(RandomNormalInitLayer, self).__init__()
        self.num_params = num_params

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.num_params,),
                                      initializer='random_normal',
                                      trainable=True)
        super(RandomNormalInitLayer, self).build(input_shape)

    def call(self, inputs):
        return self.kernel * inputs
    
def fun_PSNR(img,res):

    [M,N,L]=img.shape
    temp=1./(M*N*L)*np.sum(np.power(img-res,2))
    psnr= 10*np.log10(np.max(np.power(img,2)/temp))
    return psnr    


def residualNet(pretrained_weights = None,input_size = (256,256,1), L=10, H=0, fact = 0.5,F=0,shots_hs=1,shots_ms=1,shots_per_eval=1,y=None):
    
    inputs = Input(input_size)
    inicial = XoLayer(largo = input_size[0], ancho = input_size[1], profun = L, fact = fact)(inputs)
    
    drop1 = Dropout(0.2)(inicial)
    conv1 = Conv2D(L,3,activation='relu',use_bias=True,padding='same',kernel_initializer='he_normal')(drop1)
    conv1 = Conv2D(L,1,activation='relu',use_bias=True,padding='same',kernel_initializer='he_normal')(conv1)
    
    
    conv8 = Conv2D(L,3,activation='relu',use_bias=True,padding='same',kernel_initializer='he_normal')(conv1)
    conv8 = Conv2D(L,1,activation='relu',use_bias=True,padding='same',kernel_initializer='he_normal')(conv8)
    
    
    conv8 = Add()([conv8,conv1])
    model = Model(inputs, conv8)

    # mse = tf.keras.losses.MeanSquaredError()
    eval_idx = np.random.permutation(shots_hs+shots_ms)[:shots_per_eval]
    loss = 0
    for i in eval_idx:
        # print(y[i].shape,yp[i].shape)
        loss +=tf.reduce_mean(tf.square(y[i]-F[i](conv8)[0]))
        print(loss)
    loss = loss/shots_per_eval
    model.add_loss(loss)
        # self.add_metric(norm, name='loss',aggregation='mean')

    psnr = tf.image.psnr(inputs, conv8, 1)
    model.add_metric(psnr, name='psnr', aggregation='mean')
    return model



def UNetL(pretrained_weights = None,input_size = (256,256,1), L=10, H=0, fact = 0.5,F=0,shots_hs=1,shots_ms=1,shots_per_eval=1,y=None):
    L_2 = 2*L
    L_3 = 3*L
    L_4 = 4*L
    
    inputs = Input(input_size)  
    inicial = XoLayer(largo=input_size[0], ancho=input_size[1], profun=L, fact=fact)(inputs)
    
    conv1 = Dropout(0.2)(inicial)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)    

    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal',)(conv7)

    conv8 = Conv2D(L, 1)(conv7)

    conv8 = Add()([conv8, conv1])

    model = Model(inputs, conv8)

    # mse = tf.keras.losses.MeanSquaredError()
    eval_idx = np.random.permutation(shots_hs + shots_ms)[:shots_per_eval]
    loss = 0
    for i in eval_idx:
        # print(y[i].shape,yp[i].shape)
        loss += tf.reduce_mean(tf.square(y[i] - F[i](conv8)[0]))
        print(loss)
        
    loss2 = tf.reduce_mean(tf.square(y[i] - F[i](inputs)[0]))
    model.add_loss(loss2)   
    loss = loss / shots_per_eval
    model.add_loss(loss)
    # self.add_metric(norm, name='loss',aggregation='mean')

    psnr = tf.image.psnr(inputs, conv8, 1)
    model.add_metric(psnr, name='psnr', aggregation='mean')

    #model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def custom_loss(y, F, total_shots, shots_per_eval):
    ''' Custom loss function for SDIP approach. '''
    def loss_function(y_true, y_pred):
        loss = 0
        eval_idx = np.random.permutation(total_shots)[:shots_per_eval]
        for i in eval_idx:
            loss += tf.reduce_mean(tf.square(y[i] - F[i](y_pred)))

        loss = loss / shots_per_eval
        return loss
    return loss_function


def UNetCT(pretrained_weights=None, input_size=(256, 256, 1), L=1, d=500,F=0,
          shots_per_eval=1, y=None, total_shots=10, real_data=False):
    ''' UNet for Computed Tomography reconstruction with SDIP approach. '''

    L_2 = 2 * L
    L_3 = 3 * L
    L_4 = 4 * L

    inputs = Input(input_size)
    inicial = RandomInit(M=input_size[0],d=d)(inputs)

    conv1 = Conv2D(input_size[-1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inicial)
    conv1 = Conv2D(input_size[-1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    b_norm5 = BatchNormalization()(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(b_norm5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    b_norm6 = BatchNormalization()(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(b_norm6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    b_norm7 = BatchNormalization()(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(b_norm7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    conv8 = Conv2D(input_size[-1], 3, padding='same', activation='sigmoid')(conv7)

    model = Model(inputs, conv8)
    #model.summary()

    ssim = tf.image.ssim(conv8, inputs, max_val=1.0)
    model.add_metric(ssim, name='ssim', aggregation='mean')
    psnr = tf.image.psnr(inputs, conv8, 1)
    model.add_metric(psnr, name='psnr', aggregation='mean')


    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def Auto_encoder(pretrained_weights = None,input_size = (256,256,1), L=10, H=0, fact = 0.5,F=0,shots_hs=1,shots_ms=1,shots_per_eval=1,y=None):
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    
    inputs = Input(input_size)    
    inicial = XoLayer(largo=input_size[0], ancho=input_size[1], profun=L, fact=fact)(inputs)
    
    # the encoder part
    conv1 = Dropout(0.2)(inicial)    
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = MaxPooling2D(pool_size=(4, 4))(conv3)
    
    convup2=Conv2DTranspose(filters=L_3, kernel_size=(3, 3), strides=4, activation='relu', padding='same')(pool3)
    convup3=Conv2DTranspose(filters=L_2, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(convup2)
    conv8=Conv2DTranspose(filters=L, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(convup3)

    model = Model(inputs, conv8)

    # mse = tf.keras.losses.MeanSquaredError()
    eval_idx = np.random.permutation(shots_hs + shots_ms)[:shots_per_eval]
    loss = 0
    for i in eval_idx:
        # print(y[i].shape,yp[i].shape)
        loss += tf.reduce_mean(tf.square(y[i] - F[i](conv8)[0]))
    loss = loss / shots_per_eval
    model.add_loss(loss)
    # self.add_metric(norm, name='loss',aggregation='mean')

    psnr = tf.image.psnr(inputs, conv8, 1)
    model.add_metric(psnr, name='psnr', aggregation='mean')

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

from tensorflow.keras import backend as K

class myCallback(tf.keras.callbacks.Callback):
    def __init__(self,Xorig=0,Freq=0,shots=2):
        super(myCallback, self).__init__()
        self.my_PSNR = []
        self.Xorig = Xorig
        self.Best = np.zeros(shape=Xorig.shape)
        self.Freq = Freq
        self.shots = shots
        

    def on_epoch_end(self, epoch, logs={}):
        Freq = self.Freq
        self.model.layers[2].rate=0.0
        
        if np.mod(epoch,Freq)==0:            
            img = self.Xorig
            [m,n,L] = img.shape
           
            #func = K.function([self.model.layers[0].input],[self.model.layers[1].output])
            #xo = func(np.zeros(shape=(1,m,n,L)))
            #xo = np.asarray(xo).reshape((m,n,L),order="F")
            
            func = K.function([self.model.layers[0].input],[self.model.layers[len(self.model.layers)-(self.shots+1)].output])
            result = func(np.zeros(shape=(1,m,n,L)))
            result = np.asarray(result).reshape((m,n,L),order="F")
            
            
            psnr = fun_PSNR(img,result)
            self.my_PSNR.append(psnr) 
            print('Epoch %05d: PSNR %6.3f : Max PSNR %6.3f' % (epoch, psnr,np.max(self.my_PSNR)), flush=True)
            
            if psnr >= np.max(self.my_PSNR):                
                self.Best = result
                setattr(self.model, 'Best', self.Best)
            
            setattr(self.model, 'PSNRs', self.my_PSNR)
            
        #if np.mod(epoch,Freq*5)==0:
        #    self.model.layers[2].rate=0.5
        #    VisualGraphs(result,self.Best,xo,self.my_PSNR,img,[27,17,7])