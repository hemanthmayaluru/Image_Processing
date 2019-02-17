#Task 1.2
#import statements
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
#Function Definitions
#Function to plot the graphs
def plot_function(x_val,y_val,title,x_label,y_label):
    plt.plot(x_val, y_val, 'k-')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

#Get the initial set of values for the parameters
def initial_parameters():
    offset = 1.
    amplitude = 2.
    frequency = 16.
    phase = phi
    return offset,amplitude,frequency,phase

#Compute Fourier Transformation
def fourier_transform(o,a,freq,p,var_param):
    f = o + a * np.sin(freq*x + p)
    F = fft.fft(f)
    w = fft.fftfreq(n)
    print('Offset: ',o, 'Amplitude: ',a, 'Frequency: ',freq,'Phase: ',p)
    plot_function(w, np.abs(F),var_param,'Frequency', 'Transformed function F(X)')
	
#uni-variate function f(x) to transform
n = 512
x = np.linspace(0, 2*np.pi, n)
f = np.sin(x)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Initial Function to apply fourier transformation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#Plot the function
plot_function(x,f,'Uni-variate Function','x','sin(x)')
#compute the (discrete) Fourier transform
F = fft.fft(f)
w = fft.fftfreq(n)
#Plot the transformed function for real and imaginary part - Gets warning message as imaginary part cannot be plotted. 
#Only real part is considered for plotting
plot_function(w,F,'Fourier Transformation - Complex Number','x','F(x)')
#Plotting the transformed function - Computes the absolute(magnitude) value of the complex number
plot_function(w, np.abs(F), 'Fourier Transformation - Absolute Value','x','F(x)')
#Plotting the transformed function - log of the absolute value
plot_function(w, np.log(np.abs(F)), 'Fourier Transformation - Log of Absolute Value','x','F(x)')
#Funtion generalization :- f(x) = o + α · sin(νx + φ)
#offset values : {2,4,8}
#amplitude : {-2,+2}
#fequency : {1,8,64}
#phase : {0,π,2π}
phi = np.pi
amplitude_range = [-2,2]
frequency_range = [1,8,64]
phase_range = [0, phi, 2*phi]
offset_range = [2,4,8]

print('Varying Offset')
for offset in offset_range:
    _,amplitude,frequency,phase = initial_parameters()
    fourier_transform(offset,amplitude,frequency,phase,'Varying Offset')
print('Varying Amplitude')
for amplitude in amplitude_range:
    offset,_,frequency,phase =initial_parameters()
    fourier_transform(offset,amplitude,frequency,phase,'Varying Amplitude')
print('Varying Frequency')
for frequency in frequency_range:
    offset,amplitude,_,phase = initial_parameters()
    fourier_transform(offset,amplitude,frequency,phase,'Varying Frequency')
print('Varying Phase')
for phase in phase_range:
    offset,amplitude,frequency,_ = initial_parameters()
    fourier_transform(offset,amplitude,frequency,phase,'Varying Phase') 
'''
Observation:

Varying Frequency: With frequency increase the transformed function values move away from the zero frequency values 
                   on either side (Width increase between both the high transformed values around the zero frequency)
Varying Offset: Increasing Offset in the range of given values will double the fourier transformed function values at 
                frequency zero
Varying Amplitude: Not much effec on the output
Varying Phase: Phase variation does not have much impact on the output transformed function
'''
