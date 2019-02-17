import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys
import math
from scipy.signal import convolve, convolve2d
import time


def gaussian_kernel2D(size=15, sigma=5):
    ax = np.arange(-(size // 2 ), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)


def gaussian_kernel1D(size=15, sigma=5):
    allnum = np.linspace(-(size // 2), size // 2, size)
    gaus = np.exp(-0.5 * ((allnum / sigma) ** 2))
    return (gaus / gaus.sum())


def filter_image(img, kernel, kernelsize, imgpadded=None):
    height, width = img.shape
    # if imgpadded is None:
    imgpadded = zeropad_image(img, kernelsize)
    convolved_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            convolved_image[i, j] = convolute_matrix2D(imgpadded, kernel, i + (kernelsize // 2), j + (kernelsize // 2),
                                                       kernelsize)
    return convolved_image  # , imgpadded


def filter_image_1D(img, kernel, kernelsize, imgpadded=None):
    height, width = img.shape
    # if imgpadded is None:
    imgpadded = zeropad_image(img, kernelsize, 'col')
    convolved_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            convolved_image[i, j] = convolute_matrix1D_Row(imgpadded, kernel, i, j + kernelsize // 2, kernelsize)
    imgpadded = zeropad_image(convolved_image, kernelsize, 'row')
    convolved_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            convolved_image[i, j] = convolute_matrix1D_Col(imgpadded, kernel, i + kernelsize // 2, j, kernelsize)
    return convolved_image  # , imgpadded


def zeropad_image(imgint, kernelsize, type='all'):
    height, width = imgint.shape
    kerneloffset = kernelsize // 2
    if type == 'all':
        imgpadded = np.zeros((np.uint16(height + 2 * kerneloffset), np.uint16(width + 2 * kerneloffset)))
        imgpadded[np.uint16(kerneloffset):np.uint16(height + kerneloffset),
        np.uint16(kerneloffset):np.uint16(width + kerneloffset)] = imgint[:, :]
    elif type == 'col':
        imgpadded = np.zeros((np.uint16(height), np.uint16(width + 2 * kerneloffset)))
        imgpadded[:, np.uint16(kerneloffset):np.uint16(width + kerneloffset)] = imgint[:, :]
    elif type == 'row':
        imgpadded = np.zeros((np.uint16(height + 2 * kerneloffset), np.uint16(width)))
        imgpadded[np.uint16(kerneloffset):np.uint16(height + kerneloffset), :] = imgint[:, :]
    return imgpadded


def convolute_matrix2D(img, kernelip, x, y, k_size):
    out = 0
    koffset = k_size // 2
    for i in range(k_size):
        for j in range(k_size):
            out += img[x + koffset - i, y + koffset - j] * kernelip[i, j]
    return np.uint8(out)


def convolute_matrix1D_Row(img, kernel_ip, x, y, kernel_size):
    out = 0
    kernel_offset = kernel_size // 2
    for i in range(kernel_size):
        out += img[x, y + kernel_offset - i] * kernel_ip[i]
    return np.uint8(out)


def convolute_matrix1D_Col(img, kernel_ip, x, y, kernel_size):
    out = 0
    kernel_offset = kernel_size // 2
    for j in range(kernel_size):
        out += img[x + kernel_offset - j, y] * kernel_ip[j]
    return np.uint8(out)


def get_convolution_using_fourier_transform(image, kernel_im):
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel_im, s=image.shape, axes=(0, 1))
    transformed_image = image_fft * kernel_fft
    transformed_image = np.fft.ifft2(transformed_image)
    transformed_image = transformed_image.real
    transformed_image = transformed_image.astype(np.uint8)
    return transformed_image


if __name__ == '__main__':
    img_path = 'bauckhage.jpg'
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    task1_time = []
    task2_time = []
    task3_time = []

    imgs_method1 = []
    imgs_method2 = []
    imgs_method3 = []

    KernelSize = np.linspace(3,21,10).astype(np.uint8)
    iterations = 1
    print("Performing task 2.1")
    for i, kernelsize in enumerate(KernelSize):
        sigma = (kernelsize - 1.0) / (2.0 * 2.575)
        print("Iteration: ", i, "with kernel size", kernelsize)
        # kernelsize = 15
        timeval = 0
        kernel = gaussian_kernel2D(size=kernelsize, sigma=sigma)
        for i in range(iterations):
            start_time = time.time()
            convolutedImg = filter_image(img, kernel, kernelsize)
            end_time = time.time()
            timeval += (end_time - start_time)
        task1_time.append([timeval/iterations])
        # cv.imshow('image blurred', convolutedImg)
        imgs_method1.append(convolutedImg)
    # Task 2.2
    print("Performing task 2.2")
    for i, kernelsize in enumerate(KernelSize):
        sigma = (kernelsize - 1.0) / (2.0 * 2.575)
        print("Iteration: ", i, "with kernel size", kernelsize)
        timeval = 0
        kernel = gaussian_kernel1D(kernelsize, sigma)
        for i in range(iterations):
            start_time = time.time()
            convolutedImg = filter_image_1D(img, kernel, kernelsize)
            end_time = time.time()
            timeval += (end_time - start_time)

        task2_time.append([timeval/iterations])
        # cv.imshow('image blurred', convolutedImg)
        # cv.waitKey(1000)
        imgs_method2.append(convolutedImg)

    # Task 2.3
    print("Performing task 2.3")
    for i, kernelsize in enumerate(KernelSize):
        sigma = (kernelsize - 1.0) / (2.0 * 2.575)
        print("Iteration: ", i, "with kernel size", kernelsize)
        timeval = 0
        kernel = gaussian_kernel2D(size=kernelsize, sigma=sigma)
        for i in range(iterations):
            start_time = time.time()
            fft_result = get_convolution_using_fourier_transform(img, kernel)
            end_time = time.time()
            timeval += (end_time - start_time)
        task3_time.append([timeval/iterations])
        # cv.imshow('image blurred', fft_result)
        imgs_method3.append(fft_result)

    # cv.waitKey(0)
    x = np.arange(0, KernelSize.shape[0], 1)
    lines = plt.plot(x, task1_time, x, task2_time, x, task3_time, 'o')
    plt.setp(lines[0], linewidth=4)
    plt.setp(lines[1], linewidth=2)
    plt.setp(lines[2], markersize=10)

    plt.legend(('2D Gaussian','2 1D Gaussian', 'Using Fourier transforms' ),
               loc='upper right')
    plt.title('Time plot')
    plt.show()

    for i, kernelsize in enumerate(KernelSize):
        plt.subplot(3, 4, i+1) ,plt.imshow(imgs_method1[i], cmap = 'gray')
        plt.title('Kernel'+str(kernelsize)), plt.xticks([]), plt.yticks([])
    plt.show()

    for i, kernelsize in enumerate(KernelSize):
        plt.subplot(3, 4, i+1) ,plt.imshow(imgs_method2[i], cmap = 'gray')
        plt.title('Kernel'+str(kernelsize)), plt.xticks([]), plt.yticks([])
    plt.show()

    for i, kernelsize in enumerate(KernelSize):
        plt.subplot(3, 4, i+1) ,plt.imshow(imgs_method3[i], cmap = 'gray')
        plt.title('Kernel'+str(kernelsize)), plt.xticks([]), plt.yticks([])
    plt.show()

