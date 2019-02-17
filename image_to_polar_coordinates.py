#!/usr/bin/env python
# coding: utf-8
#@author: Hemanth Kumar Reddy Mayaluru

#imports
import numpy as np
import scipy.ndimage as img
from skimage import io
import matplotlib.pyplot as plt

#x,y to polar
def xy2rphi(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) % (2*np.pi)
    return r, phi

#polar to x,y
def rphi2xy(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x,y

#Warping to the r,phi plane
def to_r_phi_plane(f, m, n, rmax, phimax):
    rs, phis = np.meshgrid(np.linspace(0, rmax, n),np.linspace(0, phimax, m),sparse=True)
    xs, ys = rphi2xy(rs, phis)
    #origin of the x, y plane coincides with the center pixel of the given image of size m × n
    xs += n/2
    ys += m/2
    xs, ys = xs.flatten(), ys.flatten()
    coords = np.vstack((ys, xs))
    g = img.map_coordinates(f, coords, order=3)
    g = g.reshape(m, n)
    return g

#Warping to the x,y plane
def from_r_phi_plane_V2(g, m, n, rmax, phimax):
    xs, ys = np.meshgrid(np.arange(m)-m/2, np.arange(n)-n/2, sparse=True)
    rs, phis = xy2rphi(xs, ys)
    rs, phis = rs.reshape(-1), phis.reshape(-1)
    iis = phis / phimax * (m-1)
    jjs = rs / rmax * (n-1)
    coords = np.vstack((iis, jjs))
    h = img.map_coordinates(g, coords, order=3)
    h = h.reshape(m, n)
    return h

#processing the image
f = io.imread('bauckhage.jpg', as_gray=True)
m, n = f.shape
rmax = np.sqrt((m/2)**2 + (n/2)**2)
phimax = np.pi * 2
#to polar coordinates
g = to_r_phi_plane(f, m, n, rmax, phimax)
plt.imshow(g, cmap='gray')
plt.show()
#1D gaussian filter
g_blur = img.filters.gaussian_filter(g, sigma=3)
plt.imshow(g_blur, cmap='gray')
plt.show()

m_, n_ = g_blur.shape
#Back to x,y plane
h = from_r_phi_plane_V2(g_blur, m_, n_, rmax, phimax)
plt.imshow(h, cmap='gray')
plt.show()



