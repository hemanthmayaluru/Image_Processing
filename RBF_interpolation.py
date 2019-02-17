import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys
import math
from scipy.signal import convolve, convolve2d
import time
import scipy as sp
import sympy


def radial_basis_function(ip_r, sample, sigma_f):
    return np.power(math.e, -np.divide(np.power((ip_r - sample), 2), 2*np.power(sigma_f,2)))


def apply_radial_basis(ip, ipsigma):
    psi = []
    for i in range(len(ip)):
        sample = ip[i]
        psi_row = radial_basis_function(ip, sample, ipsigma)
        psi.append(psi_row)
    return np.array(psi), np.array(psi).sum(axis=1).reshape((n, 1))

n = 20
x = np.arange(n) + np.random.randn(n) * 0.2
y = np.random.rand(n) * 2
k = np.linspace(1, 5, 10)

# Sigma = [0.5, 1, 2, 4]
Sigma = [4]
for sigma in Sigma:
    psi_mat, out = apply_radial_basis(ip=x, ipsigma=sigma)
    print("The rank of the matrix is: ", np.linalg.matrix_rank(psi_mat))
    # print("The mul is", (sp.linalg.matrix_rank(psi_mat)* psi_mat))
    # print("psi mat shape (nxn): ", psi_mat.shape, "output shape", out.shape)
    psi_mat = sympy.Matrix(psi_mat)
    y = sympy.Matrix(y)
    print(y.shape, psi_mat.shape)
    inv_psi_mat = psi_mat.inv()
    print(inv_psi_mat.shape, "Is the shape of the inverse")
    w = sympy.MatMul(inv_psi_mat, y)
    w = np.array(w)
    print("The value of w is",w.shape, w)
    # sympy.Matrix.inv(psi_mat)
    # _, inds = sympy.Matrix(psi_mat).rref()  # to check the rows you need to transpose! This is now column
    # print(inds)
    # exit()
    # plt.plot(x, y, 'bo')
    # plt.plot(x, out, 'c')
    # plt.show()

    out = []
    xs = np.linspace(0,n-n/200,200)
    for lsample in xs:
        rbf_row = radial_basis_function(x, lsample, sigma)
        out.append(np.dot(rbf_row, w))
    plt.plot(x, y, 'bo')
    plt.plot(x, y, 'c')
    plt.plot(xs, out, 'k')
    # if sigma > 1:
        # plt.axis([0, 22, -5, 10])
    plt.legend(('Input Points', 'linear', 'rbf'), loc='upper right')
    plt.title('Correlation for sigma: ' + str(sigma))
    plt.show()
exit()