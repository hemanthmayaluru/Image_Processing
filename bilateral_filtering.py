import numpy as np
import cv2
import sys
import math

#Compute the bi-variate Gaussian for ⍴
def gaussian_multivariate(size, sigma):
    ax = np.arange(-(size // 2 ), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx * 2 + yy * 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)

#Compute the univariate Gaussian for σ
def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    i_filtered = 0
    Wp = 0
    gauss_multivar = gaussian_multivariate(diameter, sigma_s)
    h= int(diameter/2)
    for i in range(-h, h):
        for j in range(-h, h):
            neighbour_x = x - i
            neighbour_y = y - j
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            neighbour_x = int(neighbour_x)
            neighbour_y = int(neighbour_y)
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i) #Univariate Gaussian function
            gs = gauss_multivar[i,j] #Bi-variate Gaussian function
            w = gi * gs 
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w #Compute the normalization factor
    i_filtered = i_filtered / Wp #Apply normalization to the final image
    filtered_image[x][y] = int(round(i_filtered))


def bilateral_filter(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)
    for i in range(0, len(source)):
        for j in range(0, len(source)):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
    return filtered_image


if __name__ == "__main__":
    src = cv2.imread(str(sys.argv[1]), 0)
    filtered_image = bilateral_filter(src, 3, 10.0, 10.0)
    cv2.imwrite("filtered_image.jpg", filtered_image)