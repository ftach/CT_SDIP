import tensorflow as tf
import numpy as np

class image_prj(tf.keras.layers.Layer):
    """
    Image projection layer made of Wavelet and Radon transform.
    prj_angles: a list of angles
    """

    def __init__(self, theta=(0, 90, 180), M=512,S=512,**kwargs):
        self.theta = theta
        self.M = M
        self.S = S
        super(image_prj, self).__init__(**kwargs)

    def build(self, input_shape):
        super(image_prj, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, image_tensor):

        # Reshape the image tensor as batch, height, width, channel
        image_tensor_batched = tf.expand_dims(image_tensor, axis=-1)    
        input_shape = (self.M, self.S)

        low_tensor_batched, high_tensor_batched = self.dwt(image_tensor_batched) # compute wavelet decomposition
        #tr_high_tensor_batched = self.tf_hard_threshold(high_tensor_batched, 0) # threshold high frequency coefficients
        input_shape = (high_tensor_batched.shape[1], high_tensor_batched.shape[2]) # input shape changes after wavelet decomposition

        # APPLY RADON TRANSFORM
        tr_high_tensor = tf.reshape(high_tensor_batched, input_shape)
        padded_image_tensor, center = self.prepare_img(tr_high_tensor, input_shape) # get padded image tensor
        A = self.radon_transfom(padded_image_tensor, center) # compute Radon transform 

        return A
    
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
    
    def prepare_img(self, image_tensor, input_shape):
        ''' Pad image to avoid loose of information during Radon transform. 
        Parameters
        -------
        image_tensor: tf.tensor 
            Input data of shape (height, width) to be transformed
        input_shape: tuple
            Shape (height, width)
        Returns
        -------
        padded_image_tensor: tf.tensor 
            Padded image tensor of shape (batch, height, width, channel)
        center: int 
            Padded image center position 
        '''
        # Get dimensions
        diagonal = np.sqrt(2) * max(input_shape)
        pad = [int(np.ceil(diagonal - s)) for s in input_shape]
        new_center = [(s + p) // 2 for s, p in zip(input_shape, pad)]
        old_center = [s // 2 for s in input_shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        pad_width = tf.constant(pad_width)

        # Pad image 
        padded_image = tf.pad(image_tensor, pad_width, mode='constant', constant_values=0)
        if padded_image.shape[0] != padded_image.shape[1]:
            raise ValueError('padded_image must be a square')
        
        # Prepare image for Radon transform
        center = padded_image.shape[0] // 2
        padded_image_tensor = tf.expand_dims(padded_image, axis=0)
        padded_image_tensor = tf.expand_dims(padded_image_tensor, axis=-1)

        return padded_image_tensor, center 
    
    def tf_hard_threshold(self, image_tensor, threshold=0.075):
        ''' Apply hard thresholding to the input data.
        Parameters
        ----------
        image_tensor: tf.tensor
            Input data of shape (batch, height, width, channel) to be thresholded
        threshold: float
            Threshold value
        Returns
        -------
        thresholded_tensor: tf.tensor
            Thresholded data of shape (batch, height, width, channel)
        '''
        mask = tf.abs(image_tensor) >= threshold
        thresholded_tensor = tf.where(mask, image_tensor, tf.zeros_like(image_tensor))

        return thresholded_tensor
    
    def radon_transfom(self, padded_image_tensor, center):
        ''' Compute Radon transform of the input data.
        Parameters
        ----------
        padded_image_tensor: tf.tensor
            Input data of shape (batch, height, width, channel) to be transformed
        center: int
            Padded image center position
        Returns
        -------
        A: tf.tensor
            Sensing matrix made of wavelet decomposition and Radon transform
        '''

        radon_image = tf.zeros((padded_image_tensor.shape[1]-1, padded_image_tensor.shape[2]-1), dtype=tf.float32)
        A = []
        for i, angle in enumerate(self.theta):  # Here theta must be in radians
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            R = tf.constant([cos_a, sin_a, -center * (cos_a + sin_a - 1),
                             -sin_a, cos_a, -center * (cos_a - sin_a - 1),
                             0., 0.], dtype=tf.float32)
            R = tf.reshape(R, (1, 8))
            rotated = tf.raw_ops.ImageProjectiveTransformV3(images=padded_image_tensor, transforms=R, interpolation="BILINEAR",
                                                            output_shape=radon_image.shape, fill_mode="WRAP",
                                                            fill_value=0.0)

            radon_line = tf.reduce_sum(tf.squeeze(rotated), axis=0)
            A.append(tf.squeeze(radon_line))
        A = tf.transpose(A[:][:])

        return A/tf.reduce_max(A)

def prj(input_size=(362, 362), angles=(0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345)):
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
    prj_out = image_prj(theta=angles,M=input_size[0], S=input_size[1])(img_in)
    model = tf.keras.Model(inputs=img_in, outputs=prj_out)
    #model.summary()

    return model