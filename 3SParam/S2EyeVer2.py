import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import random

# following line enables me to print a numpy array over 1000 elements in its entirety.
np.set_printoptions(threshold=np.inf)


# creating a randomly generated input signal. 1000 samples between 0 and 100.
rng = np.random.default_rng(12345)
x = rng.integers(low=0, high=100, size=1000)
# print(x)      # Works

# This bit of code is randomly generating 20 complex numbers as S-parameters
# S_param = (rng.integers(low=0, high=100, size=20))/100 + \
#     (rng.integers(low=0, high=100, size=20))*1j/100
# print(S_param)

#                                               # Load Data

# Reads in first column of frequencies.
frequencies = np.genfromtxt("SPar_waveguide_s21.txt", dtype=None,
                            delimiter="\t", skip_header=3, usecols=0)
# print(frequencies)    # It works

# Reads in second column of real parts
Re = np.genfromtxt("SPar_waveguide_s21.txt", dtype=None,
                   delimiter="\t", skip_header=3, usecols=1)
# print(Re)      # It works

# Reads in third column of immaginary parts
Im = np.genfromtxt("SPar_waveguide_s21.txt", dtype=None,
                   delimiter="\t", skip_header=3, usecols=2)
# print(Im)       # It works.


#                                               # S-parameters and h_t

# These are transfer functions. h_t is like h(t). Sparam is like H(w). Both h_t and Sparam are 1001 long.
Sparam = Re + 1j*Im
h_t = np.fft.ifft(Sparam)


#


#                                               # Initializations

fs = 2*(330 * 1e9)      # sampling freq
# Remember, 1e2 is 100 and 10e2 is 1000
ts = 1 / fs     # sampling period

# Time Domain Signal inputs
t0 = 0      # initial time or the start time.
s_z = ts      # step size
s_c = 3000   # step count
# 3000 is a number recommended by Harry. As long as it's greater than 1001 then it's fine. Because you can convolve two signals of different lengths


#


#                                               # Bassband Signal

# below for-loop creates 3000 random bits in contiguous blocks of 30 identical bits at a time.
# e.g. [0, 0, 0, ... 0, 0, 1, 1, 1, ..., 1, 1, 0, 0, 0, ..., 0, 0, ...]
bassband = np.array([])    # Initializaing bitstream to empty numpy array.
for object in range(100):   # Repeat 100 times
    rand_b = random.randint(0, 1)   # rand_b stands for random bit.
    # random bit is repeated 30 times and concatenated onto bitstream
    bassband = np.append(bassband, (np.repeat(rand_b, 30)))

print(len(bassband))
#


#                                               # Carrier Signal

# "s_c" number of elements each between "t0" and "s_c", spaced "s_z" apart.
time = np.arange(0, s_c) * s_z + t0  # 3000 long atm
# This is correct. Check with graphics calculator
carrier = np.exp(-1j*2*fs*(np.pi)*time)


#


#                                               # Mixed Input Signal

# Element-wise multiplication of carrier and bassband, their lengths should equal
x_input = carrier * bassband


#


#                                               # Convolution and output

# x_input and h_t, which was the ifft of Sparam, is convolved to make y_output
y_output = np.convolve(x_input, h_t, mode='full')
mag_y = abs(y_output)  # calculates the magnitude of every elenent of y.

# atm, x is 3000 long and y is 4000 long.
# print("length of x is", len(x_input))
# print("length of y is", len(y_output))


#


#                                               # Plotting

# Creating x axis
tx = np.linspace(0, ts*3000, 3000)   # 2000 elements here
ty = np.linspace(0, ts*4000, 4000)   # 2000 elements here


# fig, axs = plt.subplots()  # Create a plot.
plt.plot(ty, mag_y)     # spikey one
plt.plot(tx, x_input)
plt.show()


# th = np.linspace(1, 1001, 1001)   # 2000 elements here
# plt.plot(th, abs(Sparam))   # The straight line at the top
# plt.plot(th, abs(h_t))    # The one with the sharp spike
# plt.show()
# # x is 1000 long. y is 2000 long

# print(Sparam)
# print(h_t)
