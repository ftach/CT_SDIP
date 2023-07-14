''' Computes the sensing matrix in Tensorflow such as A = RU, with R the Radon transform and U the Wavelet transfrom. 
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio 

class image_prj(tf.keras.layers.Layer):
    """
    Image projection layer made of Wavelet and Radon transform.
    prj_angles: a list of angles
    """

    def __init__(self, M=512,S=512,**kwargs):
        self.M = M
        self.S = S
        super(image_prj, self).__init__(**kwargs)

    def build(self, input_shape):
        super(image_prj, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, image_tensor):

        # Reshape the image tensor as batch, height, width, channel
        image_tensor_batched = tf.expand_dims(image_tensor, axis=1)
        image_tensor_batched = tf.expand_dims(image_tensor, axis=-1)    
        input_shape = (self.M, self.S)

        # APPLY WAVELET TRANSFORM
        low_tensor_batched, high_tensor_batched = self.dwt(image_tensor_batched) # compute wavelet decomposition
        tr_high_tensor_batched = self.tf_hard_threshold(high_tensor_batched) # threshold high frequency coefficients
        input_shape = (tr_high_tensor_batched.shape[1], tr_high_tensor_batched.shape[2]) # input shape changes after wavelet decomposition

        # APPLY FOURIER TRANSFORM
        tr_high_tensor = tf.reshape(tr_high_tensor_batched, input_shape)
        complex_tr_high_tensor = tf.complex(tr_high_tensor, tf.zeros_like(tr_high_tensor)) # convert to complex tensor
        A = self.apply_fft(complex_tr_high_tensor) 
        A = self.tf_hard_threshold(A, threshold=10) # threshold high frequency coefficients

        flat_image = tf.reshape(A, [-1, tf.shape(A)[2]])

        # Calculate the number of pixels to select
        num_pixels = tf.cast(tf.size(flat_image) * 0.125, tf.int32)

        # Randomly shuffle the pixels
        shuffled_pixels = tf.random.shuffle(flat_image)

        # Select the first 'num_pixels' pixels
        selected_pixels = tf.gather(shuffled_pixels, tf.range(num_pixels))

        # Reshape the selected pixels to the desired output shape
        output_height = tf.cast(tf.shape(image)[0] * 0.125, tf.int32)
        output_width = tf.cast(tf.shape(image)[1] * 0.125, tf.int32)
        output_image = tf.reshape(selected_pixels, [output_height, output_width, tf.shape(image)[2]])

        output_image = tf.math.real(output_image) # convert back to real tensor

        return output_image
    
    def dwt(self, image_tensor_batched):
        ''' Perfrom a 2D discrete wavelet transform on the input data. 
        Parameters
        ----------
        image_tensor_batched: tf.tensor 
            Input data of shape (batch, height, width, channel) to be transformed
        Returns
        -------
        low_pass: tf.tensor 
            Low pass of shape (batch, height, width, channel) component of the wavelet transform
        high_pass: tf.tensor
            High pass of shape (batch, height, width, channel) component of the wavelet transform
        '''

        # Haar wavelet filter 
        low_wavelet_filter = tf.constant([[0.1601, 0.6038, 0.7243, 0.1384]], dtype=tf.float32)
        low_wavelet_filter = tf.reshape(low_wavelet_filter, [2, 2, 1, 1])
        high_wavelet_filter = tf.constant([[-0.1384, 0.7243, -0.6038, 0.1601]], dtype=tf.float32)
        high_wavelet_filter = tf.reshape(high_wavelet_filter, [2, 2, 1, 1])

        # Wavelet decomposition 
        low_pass = tf.nn.conv2d(image_tensor_batched, low_wavelet_filter, strides=[1, 2, 2, 1], padding='SAME')
        high_pass = tf.nn.conv2d(image_tensor_batched, high_wavelet_filter, strides=[1, 2, 2, 1], padding='SAME')

        return low_pass, high_pass
    
    def apply_fft(self, image_tensor):
        ''' Apply FFT to image tensor.
        Parameters
        -------
        image_tensor: tf.tensor 
            Input data of shape (batch, height, width, channel)
        Returns
        -------
        fft_image_tensor: tf.tensor 
            FFT of input data of shape (batch, height, width, channel)
        '''
        fft_image_tensor = tf.signal.fft2d(image_tensor)
        fft_image_tensor = tf.signal.fftshift(fft_image_tensor)

        return fft_image_tensor
    
    def tf_hard_threshold(self, image_tensor, threshold=0.075):
        ''' Apply hard thresholding to the input data.
        Parameters
        ----------
        image_tensor: tf.tensor
            Input data of shape (batch, height, width, channel) to be thresholded
        threshold: float
            Threshold value
        Returns theta=(0, 90, 180),
        -------
        thresholded_tensor: tf.tensor
            Thresholded data of shape (batch, height, width, channel)
        '''
        mask = tf.abs(image_tensor) >= threshold
        thresholded_tensor = tf.where(mask, image_tensor, tf.zeros_like(image_tensor))

        return thresholded_tensor

def prj(input_size=(362, 362)):
    ''' Creates the Wavelet x Radon transform model
    Parameters
    ----------
    input_size: tuple
        Input image size
    angles: tuple
        Radon transform angles in radians
    Returns
    -------
    model: tf.keras.Model
        Wavelet x Radon transform model
    '''

    img_in = tf.keras.Input(input_size)
    prj_out = image_prj(M=input_size[0], S=input_size[1])(img_in)
    model = tf.keras.Model(inputs=img_in, outputs=prj_out)
    #model.summary()

    return model

# DATA LOADING
image = skio.imread('/home/paul/Documents/Florent/HDSPinternship/SDIP/brain_mri.jpg', as_gray=True)
Nt = 20
image = np.float32(image / np.max(image))
image_tensor = tf.constant(image)
image_tensor = tf.expand_dims(image_tensor, axis=0)
sz_x, sz_y = 225, 225
prj_model = prj((sz_x,sz_y))
y = prj_model.predict(image_tensor) # use tensor as input 
y_array = tf.squeeze(y).numpy()
spectrum = np.abs(y_array.flatten()**2) 
print(np.count_nonzero(spectrum)/y_array.size)
print(y_array.shape)
plt.figure()
plt.imshow(y_array, cmap = 'gray')
plt.figure()
plt.plot(spectrum)
plt.show()