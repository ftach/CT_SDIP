''' Run one Stochastic Deep Image Prior experiment for CT reconstruction on simulated data. '''

import h5py

from models.training import *  
from models.ct_wavelet import *
from models.multiple_tests_functions import *

# IMAGE DATA
img_data = h5py.File('/kaggle/input/hdspdata/ground_truth_test_000.hdf5', 'r') # MODIFY
img_dset = tf.image.resize(tf.expand_dims(img_data['data'],-1),[512,512])
image = img_dset[0, :, :]
image = np.transpose(image)
image = np.float32(image / np.max(image))

# CHOOSE PARAMETERS
NUM_ANGLES = 30
S = 10
B = 6
L = 32
input_shape = (512, 512)

# TRAINING
F, y, theta, theta_rad = get_model_inputs(image, S, NUM_ANGLES)
model, h, duration = train_model(L, B, S, F, y, image, input_shape)

#PLOTTING RESULTS
reconstruction = tf.squeeze(model(image)).numpy()
PSNR_Final = h.history['psnr']
SSIM_Final = h.history['ssim']
loss = h.history['loss']
max_PSNR = np.max(PSNR_Final)
max_SSIM = np.max(SSIM_Final)
min_loss = np.min(loss)

print('PSNR: ', max_PSNR)
print('SSIM: ', max_SSIM)
print('Loss: ', min_loss)
print('Time: ', duration)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(PSNR_Final)
plt.title('PSNR')
plt.subplot(1, 3, 2)
plt.plot(SSIM_Final)
plt.title('SSIM')
plt.subplot(1, 3, 3)
plt.plot(loss)
plt.title('Loss')

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='bone')
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(reconstruction, cmap='bone')
plt.title('Reconstruction')
plt.axis('off')

plt.show()





