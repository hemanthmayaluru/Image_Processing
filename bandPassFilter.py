#Implement Band Pass Filter

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

g = cv2.imread('bauckhage.jpg',0)   # Read the Image

G = np.fft.fft2(g)   # Compute Fourier Transformation

fshift = np.fft.fftshift(G)   
fshift2 = np.copy(fshift)

mag_spec = (np.abs(fshift))   # Calculate Amplitude Spectrum

row = G.shape[0]/2   # Number of Rows/2
colm = G.shape[1]/2   # Number of Columns/2  
 
# Take input from user
rmin = int(input('Enter rmin: '))   
rmax = int(input('Enter rmax: '))  

# Function for calculating Euclidean Distance
def dist(x1, y1, x2, y2): 
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Function for drawing Inner Circle
def make_inner_circle(tiles, row, colm, radius): 
    for x in range(row - rmin, row + rmin):
        for y in range(colm - rmin, colm + rmin):
            if dist(row, colm, x, y) <= rmin:
                fshift2[x][y] = 0

# Function for drawing Outer Circle
def make_outer_circle(tiles, row, colm, radius):
    for x in range(0, row * 2):
        for y in range(0, colm * 2):
            if dist(row, colm, x, y) > rmax:
                fshift2[x][y] = 0


make_inner_circle(fshift2, int(row), int(colm), rmin)
make_outer_circle(fshift2, int(row), int(colm), rmax)

mag_spec2 = (np.abs(fshift2))   # Calculate new Amplitude Spectrum

inv_g = np.fft.ifftshift(fshift2)   # Compute Inverse Fourier Transformation

output_img = np.real(np.fft.ifft2(inv_g))   # Reconstruct Image


plt.subplot(131),plt.imshow(g, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(mag_spec, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(mag_spec2, cmap = 'gray')
plt.title('After Suppression'), plt.xticks([]), plt.yticks([])
plt.show()
ret, th1 = cv2.threshold(output_img, 20, 255, cv2.THRESH_BINARY)
plt.figure()
plt.subplot(121),plt.imshow(g, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(th1, cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()