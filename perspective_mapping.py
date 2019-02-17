# import the necessary packages
import numpy as np
import cv2
from numpy.linalg import inv
import matplotlib.pyplot as plt


# find the pixel (x, y) on the destination image for which 
# the pixel (u, v) in source image needs to be mapped   
def pull_warp(u, v, a, b, c, d, e, f, g, h ):
    x = (int)(((a*u)+(b*v)+c)/((g*u)+(h*v)+1))
    y = (int)(((d*u)+(e*v)+f)/((g*u)+(h*v)+1))
    return x, y


# replace the pixels from the source image onto the destination image
# by finding their perspective transformation
def replace_pixels(destination, source):
    a, b, c, d, e, f, g, h = find_coefficients(source)
    for u in range(0, source.shape[0]):
        for v in range(0, source.shape[1]):
            x, y = pull_warp(u, v, a, b, c, d, e, f, g, h )
            destination[x][y] = source[u][v]
    return destination


def find_coefficients(input_im):
    # intitialize the coordinates for source and destination images
    x1 = 56
    y1 = 215
    x2 = 10
    y2 = 365
    x3 = 258
    y3 = 218
    x4 = 296
    y4 = 364
    u1 = 0
    v1 = 0
    u2 = 0
    v2 = input_im.shape[1]
    u3 = input_im.shape[0]
    v3 = 0
    u4 = input_im.shape[0]
    v4 = input_im.shape[1]

    z1 = -u1*x1
    z2 = -u2*x2
    z3 = -u3*x3
    z4 = -u4*x4
    z5 = -u1*y1
    z6 = -u2*y2
    z7 = -u3*y3
    z8 = -u4*y4

    z9 = -v1*x1
    z10 = -v2*x2
    z11 = -v3*x3
    z12 = -v4*x4
    z13 = -v1*y1
    z14 = -v2*y2
    z15 = -v3*y3
    z16 = -v4*y4
   
    A = [[u1, v1, 1, 0, 0, 0, z1, z9],
        [u2, v2, 1, 0, 0, 0, z2, z10],
        [u3, v3, 1, 0, 0, 0, z3, z11],
        [u4, v4, 1, 0, 0, 0, z4, z12],
        [0, 0, 0, u1, v1, 1, z5, z13],
        [0, 0, 0, u2, v2, 1, z6, z14],
        [0, 0, 0, u3, v3, 1, z7, z15],
        [0, 0, 0, u4, v4, 1, z8, z16]]
   
    B = [[x1], [x2], [x3], [x4], [y1], [y2], [y3], [y4]]

    print(np.linalg.matrix_rank(A))
    Ainv = inv(A)
    M = np.matmul(Ainv, B)   
   
    # result of matrix multiplication gives the values a, b, ... , h
    return (M[0], M[1], M[2],  M[3],  M[4], M[5], M[6], M[7])


if __name__ == "__main__":
    destination = cv2.imread("isle.jpg")
    source = cv2.imread("bauckhage.jpg")

    warped = replace_pixels(destination, source)
    cv2.imwrite("newimage.jpg", warped)
    cv2.imshow("warpedImage.jpg", warped)
    cv2.waitKey(0)
