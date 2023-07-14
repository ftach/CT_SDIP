''' Run different tests changing parameters values.'''
import numpy as np
import os
import tensorflow as tf
import pandas as pd
# import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import time
import h5py
import scipy
from models.training import *  
from models.ct_wavelet import *
from models.multiple_tests_functions import *

# MANAGING GPU MEMORY
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=8 * 1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

results = pd.DataFrame(columns=['img_nbr', 'L', 'B', 'max_PSNR', 'max_SSIM', 'Time'])

# CHOOSE TESTED IMAGES
all_img = []
img_data = h5py.File('/kaggle/input/hdspdata/ground_truth_test_000.hdf5', 'r') # MODIFY
img_dset = tf.image.resize(tf.expand_dims(img_data['data'],-1),[512,512])
index = [1, 4, 9, 14, 20] # MODIFY
for i in index:
    image = img_dset[i, :, :]
    image = np.transpose(image)
    image = np.float32(image / np.max(image))
    all_img.append(image)

# CHOOSE TESTED PARAMETERS
results.to_csv('angles_results.csv', index=False)
img_nbr = 0
L = 32
# all_L = [4, 16, 32, 64, 96, 128]
#all_S = [10, 20, 30]
angles_distribution = ['linear', 'random', 'gaussian']
S = 10
B = 6
input_shape = (512, 512)
nb_angles = 30
for image in all_img:
    img_nbr += 1
    #for distrib in angles_distribution:
    distrib = 'uniform'
        # step = S//5
        #all_B = range(2, S, 2)
        #for B in all_B:
        #for nb_angles in range(10, 61, 10):
    for i in range(5):  # 5 try for each image
        F, y, theta, theta_rad = get_model_inputs(image, S, distrib, NUM_ANGLES=nb_angles)
        print('start training')
        model, h, duration = train_model(L, B, S, F, y, input_shape)
        # reconstruction = tf.squeeze(model(image)).numpy()
        # plt.imsave('reconstruction'+str(img_nbr)+'.png', reconstruction, cmap='gray')
        PSNR_Final = h.history['psnr']
        SSIM_Final = h.history['ssim']
        # loss = h.history['loss']
        max_PSNR = np.max(PSNR_Final)
        max_SSIM = np.max(SSIM_Final)
        df2 = pd.DataFrame(
            {'S': [S], 'img_nbr': [img_nbr], 'L': [L], 'B': [B], 'max_PSNR': [max_PSNR], 'max_SSIM': [max_SSIM],
             'Time': [duration], 'distribution': [distrib], 'NUM_ANGLES': [nb_angles]})
        results = pd.concat([results, df2], ignore_index=True)
        results.to_csv('angles_results.csv', index=False)

