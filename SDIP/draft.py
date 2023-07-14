'''Draft python file to open a DICOM file and compute the golden angles '''
import pydicom 
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon

# Read the DICOM file
ds = pydicom.dcmread("/home/paul/Documents/Florent/CT_SDIP/SDIP/L632_4L_120kv_quarterdose1.00001.dcm") # MODIFY
proj = ds.pixel_array 
print(ds)
ri = ds.RescaleIntercept
rs = ds.RescaleSlope
proj = proj * rs + ri
print(proj.shape)
proj = np.transpose(proj)
theta = np.linspace(0., 180., proj.shape[1], endpoint=False)
img = iradon(proj, theta=theta, circle=False)
plt.imshow(img, cmap='gray')
plt.show()


# COMPUTE GOLDEN ANGLES
def golden_angles(NUM_ANGLES, max_value=360):
    theta = []
    for k in range(NUM_ANGLES):
        angle = k*111.25 % max_value
        theta.append(angle)
    return theta

gold_angles = golden_angles(30, 180)
print(gold_angles)
angles = [i*180/513 for i in range(513)]

 

