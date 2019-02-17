#Phase and Magnitude

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

#no of samples
n = 512

#A function that plots FFT of the input 
def plotFFT(f, x, title):
        #compute the (discrete) Fourier transform F(w) of f(x)
        F = fft.fft(f)
        #get the frequencies
        w = fft.fftfreq(n)
        #plot f(x) and the FFT
        plt.subplot(121)
        plt.plot(x, f, 'k-')
        plt.title(title)
        plt.subplot(122)
        plt.plot(w, np.abs(F), 'k-') #Taking absolute values as F is a complex valued function
        plt.title("Fourier Transform")
        plt.show()

#create a uni-variate function f(x) to transform
x = np.linspace(0, 2*np.pi, n)
f = np.sin(x)
#plot the Fourier Transform of f(x)
plotFFT(f, x, "f(x) = sin(x)")
#generalize the function f(x)
offset = 1
amplitude = 2
frequency = 16
phase = np.pi
#create a new function f(x) to transform
f_new = offset + amplitude * np.sin(frequency*x + phase)
#plot the Fourier Transform of f_new
plotFFT(f_new, x, "f(x) = o + A*sin(wx + p)")

#the signal is translated from time domain to frequency domain