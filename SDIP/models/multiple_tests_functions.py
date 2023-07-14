''' Functions that enable the training of the UNetCT model with different parameters '''

import numpy as np
import time 
from models.training import *  
from models.ct_wavelet import *
from skimage.transform import resize

def get_real_model_inputs(image, sin, S, real_data=False, NUM_ANGLES=30):
    '''
    Parameters
    ----------
    image: tf.tensor
        Input image of shape (batch, height, width, channel)
    sin: tf.tensor
        Sinogram of the input image
    S: int
        Total number of shots
    NUM_ANGLES: int 
        Number of angles for the Radon transform
    '''
    _, sz_x, sz_y = image.shape
    F = []
    y =[]
    theta = []
    theta_rad = []
    for i in range(S):
        #indexes = np.random.normal(loc=sin.shape[1]/2, scale=sin.shape[1]/2, size=NUM_ANGLES )
        indexes = np.random.uniform(0, sin.shape[1], size=NUM_ANGLES)
        #indexes = np.random.poisson(sin.shape[1]/2, size=NUM_ANGLES)
        #indexes = np.linspace(0, sin.shape[1], NUM_ANGLES, endpoint=False)
        #indexes = indexes.clip(0, sin.shape[1]-1)
        indexes = sorted(indexes.astype(int))
        angles = [x*180/sin.shape[1] for x in indexes]
        angles_rad = [(x* np.pi / 180) for x in angles]
        theta.append(angles)
        theta_rad.append(angles_rad)    
        proj = sin[:, indexes]
        proj = resize(proj, (362, NUM_ANGLES))
        F.append(prj((sz_x, sz_y), theta_rad[i]))
        sim_proj = F[i](image)
        proj = np.float32((proj-np.min(proj)) / (np.max(proj) - np.min(proj))) 
        proj *= np.max(sim_proj)
        if real_data:
            y.append(proj)
        else:
            y.append(sim_proj)

    return F, y, theta, theta_rad

def train_real_model(L, B, S, F, y, image, input_shape, lr=0.001, iters=10000): 
    '''
    Parameters
    ----------
    L: int
        Number of layers
    B: int
        Number of shots per evaluation
    S: int
        Total number of shots
    F: list
        List of forward operators
    y: list
        List of measurements 
    image: tf.tensor
        Input image of shape (batch, height, width)
    input_shape: tuple
        Input image shape (height, width)
    lr: float
        Learning rate
    iters: int
        Number of iterations
    Returns
    -------
    model: tf.keras.Model
        Trained model
    h: tf.keras.callbacks.History
        History of the training
    duration: float
        Training time in seconds
    '''
    model = UNetCT(input_size=(input_shape[0],input_shape[1],1), F=F, y=y, L=L, shots_per_eval=B,d=400, total_shots=S, real_data=True)
    optimizad = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.99, amsgrad=False)
    model.compile(optimizer=optimizad, loss=custom_loss(y, F, S, B))
    start = time.time()
    h = model.fit(x=image,y=image, epochs=iters, batch_size=1, verbose=1)
    end = time.time()
    duration = end - start

    return model, h, duration 

def get_model_inputs(image, S, NUM_ANGLES=30):
    '''
    Parameters
    ----------
    image: tf.tensor
        Input image of shape (batch, height, width, channel)
    S: int
        Total number of shots
    NUM_ANGLES: int
        Number of angles for the Radon transform

    Returns
    -------
    F: list
        List of Fourier transform of the input image
    y: list
        List of Radon transform of the input image
    theta: list
        List of Radon transform angles
    theta_rad: list
        List of Radon transform angles in radians
    '''
    _, sz_x, sz_y = image.shape
    F = []
    y =[]
    theta = []
    theta_rad = []
    for i in range(S) :
        angles = np.random.normal(loc=90, scale=90, size=NUM_ANGLES)
        rad_angles = [(x* np.pi / 180) for x in angles] 
        angles = sorted(angles)
        rad_angles = sorted(rad_angles)
        theta.append(angles) 
        theta_rad.append(rad_angles)
        F.append(prj((sz_x, sz_y), theta_rad[i]))
        y.append(F[i](image))

    return F, y, theta, theta_rad

def train_model(L, B, S, F, y, image, input_shape, lr=0.001, iters=10000): 
    '''
    Parameters
    ----------
    L: int
        Number of layers
    B: int
        Number of shots per evaluation
    S: int
        Total number of shots
    F: list
        List of forward operators
    y: list
        List of measurements 
    image: tf.tensor
        Input image of shape (batch, height, width)
    input_shape: tuple
        Input image shape (height, width)
    lr: float
        Learning rate
    iters: int
        Number of iterations
    Returns
    -------
    model: tf.keras.Model
        Trained model
    h: tf.keras.callbacks.History
        History of the training
    duration: float
        Training time in seconds
    '''

    model = UNetCT(input_size=(input_shape[0],input_shape[1],1), F=F, y=y, L=L, shots_per_eval=B,d=400, total_shots=S)
    optimizad = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.99, amsgrad=False)
    model.compile(optimizer=optimizad, loss='mean_squared_error')
    start = time.time()
    h = model.fit(x=image,y=image, epochs=iters, batch_size=1, verbose=0)
    end = time.time()
    duration = end - start

    return model, h, duration 