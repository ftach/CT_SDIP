
''' Computes the sensing matrix in Tensorflow such as A = RU, with R the Radon transform and U the DCT2D transfrom. 
'''
import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.transform import iradon

import tensorflow as tf


class image_prj(tf.keras.layers.Layer):
    """
    Image projection layer.
    prj_angles: a list of angles
    """

    def __init__(self, theta=(0, 90, 180), M=512,S=512,**kwargs):
        self.theta = theta
        self.M = M
        self.S = S
        super(image_prj, self).__init__(**kwargs)

    def build(self, input_shape):
        super(image_prj, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, image, input_shape=(362, 362)):

        #image_tensor_batched = tf.expand_dims(image_tensor, axis=0)  # Reshape the image tensor to add a batch dimension
        image_tensor_batched = tf.expand_dims(image,
                                              axis=0)  # Reshape the image tensor to add a batch dimension

        dct_tensor = self.dct_2d(image_tensor_batched, norm='ortho')
        #dct_tensor = self.dct_2d(dct_tensor, norm='ortho')
        image = tf.reshape(dct_tensor, (self.M, self.M))

        diagonal = np.sqrt(2) * max(input_shape)
        pad = [int(np.ceil(diagonal - s)) for s in input_shape]
        new_center = [(s + p) // 2 for s, p in zip(input_shape, pad)]
        old_center = [s // 2 for s in input_shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]

        pad_width = tf.constant(pad_width)

        padded_image = tf.pad(image, pad_width, mode='constant', constant_values=0)

        if padded_image.shape[0] != padded_image.shape[1]:
            raise ValueError('padded_image must be a square')
        center = padded_image.shape[0] // 2
        radon_image = tf.zeros((padded_image.shape[0], padded_image.shape[1]), dtype=tf.float32)
        padded_image = tf.expand_dims(padded_image, axis=0)
        padded_image = tf.expand_dims(padded_image, axis=-1)
        image = tf.expand_dims(image, axis=0)
        image = tf.expand_dims(image, axis=-1)  #

        A = []
        for i, angle in enumerate(self.theta):  # Here theta must be in radians
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            R = tf.constant([cos_a, sin_a, -center * (cos_a + sin_a - 1),
                             -sin_a, cos_a, -center * (cos_a - sin_a - 1),
                             0., 0.], dtype=tf.float32)
            R = tf.reshape(R, (1, 8))
            rotated = tf.raw_ops.ImageProjectiveTransformV3(images=padded_image, transforms=R, interpolation="BILINEAR",
                                                            output_shape=radon_image.shape, fill_mode="WRAP",
                                                            fill_value=0.0)

            radon_line = tf.reduce_sum(tf.squeeze(rotated), axis=0)
            A.append(tf.squeeze(radon_line))
        y = tf.transpose(A[:][:])
        return y/tf.reduce_max(y)


    def dct_2d(self,feature_map, norm=None):
        ''' Computes the 2D DCT transform of a feature map'''
        feature_map = tf.expand_dims(tf.squeeze(feature_map), axis=0)
        # print(feature_map.shape)
        X1 = tf.transpose(tf.signal.dct(feature_map, type=2, norm=norm), perm=[0, 2, 1])
        X2 = tf.signal.dct(X1, type=2, norm=norm)
        X2_t = tf.transpose(X2, perm=[0, 2, 1])

        return X2_t


def idct_2d(feature_map, norm=None):
    ''' Computes the 2D IDCT transform of a feature map'''

    X1 = tf.signal.idct(feature_map, type=2, norm=norm)
    X1_t = tf.transpose(X1, perm=[0, 1, 3, 2])
    X2 = tf.signal.idct(X1_t, type=2, norm=norm)
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])

    return X2_t


def prj(input_size=(192, 192, 1), angles=(0, 1, 2, 3, 4, 5, 6, 7)):
    ''' Creates the Radon transform model'''
    img_in = tf.keras.Input(input_size)
    prj_out = image_prj(theta=angles,M=input_size[0])(img_in)
    model = tf.keras.Model(inputs=img_in, outputs=prj_out)
    # model.summary()
    return model
