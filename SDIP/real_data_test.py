''' Run tests on real data for CT reconstruction. '''
import h5py
import numpy as np
from models.training import *
from models.multiple_tests_functions import *
from models.ct_wavelet import *
import pandas as pd 


# MANAGING GPU MEMORY
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8*1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

results = pd.DataFrame(columns=['img_nbr', 'L', 'B', 'max_PSNR', 'max_SSIM', 'Time', 'NUM_ANGLES', 'S'])

sin_data = h5py.File('observation_test_000.hdf5', 'r') # MODIFY
img_data = h5py.File('ground_truth_test_000.hdf5', 'r') # MODIFY
sin_dset = sin_data['data'] 
img_dset = tf.image.resize(tf.expand_dims(img_data['data'],-1),[512,512])

chosen_img = [1, 2, 3, 4, 5]
all_img = []
for i in chosen_img:
    sin = sin_dset[i, :, :]
    img = img_dset[i, :, :]
    sin = np.transpose(sin)
    img = np.transpose(img)
    all_img.append((img, sin))

all_S = [10, 20]
all_L = [32, 64]
img_nbr = 0
input_shape = (512, 512)
for images in all_img:
    img_nbr += 1
    image = images[0]
    sin = images[1]
    for S in all_S:
        if S == 10:
            B = 6
            all_angles = [10, 20]
        else:
            B = 9
            all_angles = [5, 10]
        for NUM_ANGLES in all_angles:
            for L in all_L:
                F, y, theta, theta_rad = get_real_model_inputs(image, sin, S, real_data=False, NUM_ANGLES=NUM_ANGLES)
                model, h, duration = train_real_model(L, B, S, F, y, image, input_shape)
                reconstruction = tf.squeeze(model(image)).numpy()
                PSNR_Final = h.history['psnr']
                SSIM_Final = h.history['ssim']
                loss = h.history['loss']
                if img_nbr == 0 or img_nbr == 1: 
                    plt.imsave('reconstruction'+str(img_nbr)+str(NUM_ANGLES)+str(S)+'.png', reconstruction, cmap='bone')
                df2 = pd.DataFrame({'img_nbr': [img_nbr], 'L': [L], 'B': [B], 'max_PSNR': [np.max(PSNR_Final)], 'max_SSIM': [np.max(SSIM_Final)], 'Time': [duration], 'NUM_ANGLES': [NUM_ANGLES], 'S': [S]})
                results = pd.concat([results, df2], ignore_index=True)

results.to_csv('results.csv', index=False) 
