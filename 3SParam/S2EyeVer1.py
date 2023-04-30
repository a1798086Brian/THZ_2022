# Version 1 was flawed and discontinuted
# Written by Brian Wang
# Discontinued on 27/01/2022

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# creating a randomly generated input signal. 1000 samples between 0 and 100.
rng = np.random.default_rng(12345)
x = rng.integers(low=0, high=100, size=1000)
# print(x)      # Works

# This bit of code is randomly generating 20 complex numbers as S-parameters
# S_param = (rng.integers(low=0, high=100, size=20))/100 + \
#     (rng.integers(low=0, high=100, size=20))*1j/100
# print(S_param)


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


# The transfer functions.
# h_t is like h(t). Sparam is like H(w)
# both h_t and Sparam are 1001 long.
Sparam = Re + 1j*Im
h_t = np.fft.ifft(Sparam)


# Computing output
y = np.convolve(x, h_t, mode='full')
mag_y = abs(y)  # calculates the magnitude of every elenent of y.


# Creating x axis
tx = np.linspace(1, 1000, 1000)   # 2000 elements here
ty = np.linspace(1, 2000, 2000)   # 2000 elements here


# # fig, axs = plt.subplots()  # Create a plot.
# plt.plot(ty, mag_y)
# plt.plot(tx, x)
# plt.show()
# # x is 1000 long. y is 2000 long

th = np.linspace(1, 1001, 1001)   # 2000 elements here
plt.plot(th, abs(Sparam))   # The straight line at the top
plt.plot(th, abs(h_t))    # The one with the sharp spike
plt.show()
# x is 1000 long. y is 2000 long

print(Sparam)
print(h_t)
