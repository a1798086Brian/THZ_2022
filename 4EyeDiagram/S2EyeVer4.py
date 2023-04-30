# pylint: disable=C0103

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


#


#                                               # Load Data

# Reads in first column of frequencies.
freq = np.genfromtxt("SPar_waveguide_s21.txt", dtype=None,
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


#


#                                               # S-parameters and h_t

# These are transfer functions. h_t is like h(t). Sparam is like H(w). Both h_t and Sparam are 1001 long.
Sparam = Re + 1j*Im     # BTW Sparam is basically S21

# work out what is the increment in each freq jump. "freq_s_z" stands for frequency step size
freq_s_z = (np.max(freq) - np.min(freq)) / (len(freq) - 1)

# The below code adds 2000 zeroes before Sparam.
# How I got 2000 is weird. It's because 220GHz to 330Hz requires 1001 numbers. 1001 not 1000 because you have to add an extra...
# ... number when you include both ends. E.g. 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 is actually 11 numbers. But, if we...
# ... are going from 0 to 220, including 0 but not including 220, then we would have 2000 numbers. If we included 220, then...
# ... we would have 2001 numbers.
zeroes = np.zeros(2000)
# 0s first, then Sparam. In the order of 0-220, then 220-330
Sparam = np.append(zeroes, Sparam)

# Now we want to add 3001 or 3000 (both are probably fine) to the end of Sparam. This is represents the negative part of the...
# ... frequencies Numpy for some reason wants after the postive part.
zeroes2 = np.zeros(len(Sparam))
Sparam = np.append(Sparam, zeroes2)

# ifft to find the Sparam in time domain
h_t = np.fft.ifft(Sparam)


#


#                                               # Initializations

# sampling freq     # This is 2 times the maximum frequency.
fs = 2*(330 * 1e9)
# Remember, 1e2 is 100 and 10e2 is 1000
ts = 1 / fs     # sampling period

# Time Domain Signal inputs
t0 = 0      # initial time or the start time.
s_z = ts      # step size
s_c = 33000   # step count
# 3000 is a number recommended by Harry. As long as it's greater than 1001 then it's fine. Because you can convolve two signals of different lengths

# fb is the frequency of the bassband signal    # How fast we are transimitting data. 2GHz == 2Gbit
fb = 2*(1e9)    # Can go up to 30Gbit

# symbol_len is the length of each symbol in number of bits
symbol_len = 1/(fb * ts)
symbol_count = 100  # this is how many symbols we want in our bass band signal


#


#                                               # Bassband Signal

# below for-loop creates 3000 random bits in contiguous blocks of 30 identical bits at a time.
# e.g. [0, 0, 0, ... 0, 0, 1, 1, 1, ..., 1, 1, 0, 0, 0, ..., 0, 0, ...]
bassband = np.array([])    # Initializaing bitstream to empty numpy array.
for object in range(symbol_count):   # Repeat 100 times
    rand_b = random.randint(0, 1)   # rand_b stands for random bit.
    # random bit is repeated 30 times and concatenated onto bitstream
    bassband = np.append(bassband, (np.repeat(rand_b, symbol_len)))


#


#                                               # Carrier Signal

# "s_c" number of elements each between "t0" and "s_c", spaced "s_z" apart.
time = np.arange(0, s_c) * s_z + t0  # 3000 long atm
# This is correct. Check with graphics calculator
# This is the freq the waveguide works at, goes from 220GHz to 330GHz
carrier_fs = 275 * 1e9
carrier = np.exp(1j*2*carrier_fs*(np.pi)*time)


#


#                                               # Mixed Input Signal

# Element-wise multiplication of carrier and bassband, their lengths should equal
x_input = carrier * bassband
mag_x = abs(x_input)


#


#                                               # Convolution and output

# x_input and h_t, which was the ifft of Sparam, is convolved to make y_output
y_output = np.convolve(x_input, h_t, mode='full')
mag_y = abs(y_output)  # calculates the magnitude of every elenent of y.

# atm, x is 3000 long and y is 4000 long.
# print("length of x is", len(x_input))
# print("length of y is", len(y_output))

# print(len(x_input))
# print(len(y_output))


#


# COMMENT ON/OFF START


#                                         # Plotting 1 - input and output in time domain

# Creating x axis and y axis for time domain plots
tx = np.linspace(0, ts*(len(mag_x)), len(mag_x))   # 2000 elements here
ty = np.linspace(0, ts*(len(mag_y)), len(mag_y))   # 2000 elements here


# fig, axs = plt.subplots()  # Create a plot.
plt.plot(ty, mag_y)     # spikey one
plt.plot(tx, mag_x)     # boxy one
plt.legend(["output_y", "input_x"])
plt.title('Input and Output Signals in the Time Domain')
plt.show()


#                                       # Plotting 2 - all input signals in frequency domain
# plotting fft of each input signals
bassband_fft = np.fft.fft(bassband)
carrier_fft = np.fft.fft(carrier)
x_input_fft = np.fft.fft(x_input)

plt.figure()
plt.plot(abs(bassband_fft))
plt.plot(abs(carrier_fft))
plt.plot(abs(x_input_fft))
plt.legend(["bassband_fft", "carrier_fft", "x_input_fft"])
plt.title('DFT of Bassband, Carrier, and x_input Signals Respectively')
plt.show()


# COMMENT ON/OFF END


#


#                                               # Testing Code

# th = np.linspace(1, 1001, 1001)   # 2000 elements here
# plt.plot(th, abs(Sparam))   # The straight line at the top
# plt.plot(th, abs(h_t))    # The one with the sharp spike
# plt.show()
# # x is 1000 long. y is 2000 long

# print(Sparam)
# print(h_t)


#


#                                               # Eye Diagram Plotting

# parameters
tb = 1/fb   # tb is length of each symbol. Note, bassband is not sinusoidal.
# fb is the frequency of the bassband signal in Hz. fb was previously defined earlier in the code.

eye_diag_window = 3*tb  # arbitrary, we typically display 3 tb in eye diagram
marker_size = 1  # how big the markers are on the graph.


# Setting up subplots
# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ax = plt.subplots(1, 2)
# fig.suptitle('Horizontally stacked subplots')
# ax1.scatter([1, 2, 3, 4], [1, 2, 3, 4])
# ax1.scatter([1, 2, 3, 4], [-1, -2, -3, -4])
# ax2.scatter([1, 2, 3, 4], [-1, -2, -3, -4])


#                                         # Time Domain (Eye diagram)
ax[0].scatter(np.mod(tx, eye_diag_window), mag_x, s=marker_size)
ax[0].scatter(np.mod(ty, eye_diag_window), mag_y, s=marker_size)

th = np.linspace(0, ts*(len(h_t)), len(h_t))   # 2000 elements here
ax[0].scatter(np.mod(th, eye_diag_window), abs(h_t), s=marker_size)
ax[0].set_title('Time Domain')
ax[0].legend(["Input Signal", "Output Signal", "ifft($S_{21}$)"])
ax[0].set_ylabel("Amplitude (a.u.)")
ax[0].set_xlabel("Seconds (s)")


#                                         # Frequency Domain

sample_frequencies_Sparam = np.fft.fftfreq(len(Sparam), d=(1/(2*max(freq))))
sample_frequencies_carrier_fft = np.fft.fftfreq(
    len(carrier_fft), d=(1/(2*max(freq))))

ax[1].plot(sample_frequencies_carrier_fft, abs(
    carrier_fft)/(np.max(abs(carrier_fft))))
ax[1].plot(sample_frequencies_Sparam, abs(Sparam))
# ax[0, 0].set_title("Sine function")
ax[1].legend(["275 GHz Carrier", "$S_{21}$"])

# Remember Sparam is already in freq domain
ax[1].set_title('Frequency Domain')
ax[1].set_xlabel("Frequency (GHz)")

plt.show()


print(len(carrier_fft))
print(len(Sparam))
# print(mag_y)


#   fftfreq testing ground and others
testcase = np.fft.fftfreq(1000, d=(1/(2*max(freq))))
# print(testcase)

print(max(freq))
# print(len(ty))
# print(len(mag_y))


# print(len(h_t))


# print(np.max(abs(carrier)))
