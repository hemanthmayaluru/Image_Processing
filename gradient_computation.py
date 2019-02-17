import numpy as np
import scipy.stats as st
import cv2
from numpy import pi, exp, sqrt
from scipy import ndimage
from matplotlib import pyplot as plt

#------------------------------------Function Definitions- Start----------------------------------

# Normalized Kernel

def kernel2d(sig,k):
    # First a 1-D  Gaussian
    t = np.linspace(-sig, sig, k)
    bump = np.exp(-0.1*t**2)
    bump /= np.trapz(bump) # normalize the integral to 1

    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    return kernel

#------------------------------------Function Definitions- End----------------------------------

#Loading Images
img_clock = cv2.imread('clock.jpg',0)
img_bauckhage = cv2.imread('bauckhage.jpg',0)
#Generate Gaussian Kernel
kernel = kernel2d(2.0,7)
#Compute Gradient of Kernel
kernelX,kernelY = np.gradient(kernel)

#Image Gradients
gradXImg = ndimage.convolve(img_bauckhage, kernelX)
gradYImg = ndimage.convolve(img_bauckhage, kernelY)
#Magnitude of the image gradient
magnitude_gradient = np.sqrt(np.add(np.square(gradXImg),np.square(gradYImg)))
plt.imshow(magnitude_gradient.astype('uint8'), cmap="Greys_r")
plt.show()
#Clock
gradXImg_clock = ndimage.convolve(img_clock, kernelX)
gradYImg_clock = ndimage.convolve(img_clock, kernelY)
magnitude_gradient_clock = np.sqrt(np.add(np.square(gradXImg_clock),np.square(gradYImg_clock)))
plt.imshow(magnitude_gradient_clock.astype('uint8'), cmap="Greys_r")
plt.show()





