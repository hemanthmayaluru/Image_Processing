import numpy as np
from skimage.io import imread
from skimage.transform import PiecewiseAffineTransform, warp
import matplotlib.pylab as plt

alpha1 = 15
alpha2 = 15
v1 = 1.5
v2 = 1
phi1 = np.pi/8
phi2 = np.pi/2

def wave(xy):
    xy[:, 0] += alpha1*np.sin(v1*np.pi*xy[:, 1]/32 - phi1)
    xy[:, 1] += alpha2*np.sin(v2*np.pi*xy[:, 0]/32 -phi2)
    return xy

def wave_x(image):
    rows, cols = image.shape[0], image.shape[1]
    src_cols = np.linspace(0, cols, 40)
    src_rows = np.linspace(0, rows, 40)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    alpha = 50 
    v = 1/4 
    phi = -np.pi*3/4
    # add sinusoidal oscillation to row coordinates
    dst_rows1 = src[:, 1] - np.sin(v*np.linspace(0, 6 * np.pi, src.shape[0]) - phi) * alpha
    dst_cols1 = src[:, 0]
    dst_rows1 *=1.5
    dst_rows1 -= alpha*1.5
    dst1 = np.vstack([dst_cols1, dst_rows1]).T
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst1)
    return  warp(image, tform, output_shape=(rows, cols))


def swirl(xy, x0, y0, R):
    r = np.sqrt((xy[:,1]-x0)**2 + (xy[:,0]-y0)**2)
    a = np.pi * r / R
    xy[:, 1] = (xy[:, 1]-x0)*np.cos(a) + (xy[:, 0]-y0)*np.sin(a) + x0
    xy[:, 0] = -(xy[:, 1]-x0)*np.sin(a) + (xy[:, 0]-y0)*np.cos(a) + y0
    return xy

im = imread('clock.jpg', as_gray=True)
im1 = warp(im, wave)
plt.imshow(im1, cmap = 'gray')
plt.show()

im2 = wave_x(im)
plt.imshow(im2, cmap = 'gray')
plt.show()

#image -1 : funtion used: wave_x, alpha = 50 v = 1/4 phi = -pi*3/4 dst_rows1 -= alpha*1.5
#image -2 : funtion used: wave_x, alpha = 60 v = 5/16 phi = pi dst_rows1 -= alpha*3/4
'''Image -3: funtion used: wave
alpha1 = 15
alpha2 = 10
v1 = 0.5
v2 = 0.5
phi1 = np.pi/8
phi2 = np.pi/2'''
''' Image - 4: funtion used: wave 
alpha1 = 15
alpha2 = 15
v1 = 1.5
v2 = 1
phi1 = np.pi/8
phi2 = np.pi/2'''
'''Image - 5: funtion used: wave
alpha1 = 15
alpha2 = 20
v1 = 0.25
v2 = 2
phi1 = -np.pi
phi2 = 0'''