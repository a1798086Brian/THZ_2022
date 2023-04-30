# pylint: disable=C0103

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random

# following line enables me to print a numpy array over 1000 elements in its entirety.
np.set_printoptions(threshold=np.inf)


#


#                       # Load Data

# Reads in first column of frequencies.
freq = np.genfromtxt("SPar_waveguide_s21.txt", dtype=None,
                     delimiter="\t", skip_header=3, usecols=0)

# Reads in second column of real parts
Re = np.genfromtxt("SPar_waveguide_s21.txt", dtype=None,
                   delimiter="\t", skip_header=3, usecols=1)

# Reads in third column of immaginary parts
Im = np.genfromtxt("SPar_waveguide_s21.txt", dtype=None,
                   delimiter="\t", skip_header=3, usecols=2)


#


#                       # S-parameters and h_t

# These are transfer functions. h_t is like h(t). S21 is like H(w). Both h_t and S21 are 6002 long.
S21 = Re + 1j*Im     # S21 is basically the Scattering parameters

# This line dds 2000 zeroes before S21. The 2000 zeroes means there were no S-parameters for 0-220 Hz. In reality there is,
# ... we just can't picked it up. Why 2000 zeroes? See appendix for further explanation.
zeroes = np.zeros(2000)
S21 = np.append(zeroes, S21)    # 0s first, then S21. 0-220Hz, then 220-330Hz

# Now add 3001 or 3000 (both are probably fine) to the end of S21. This is represents -330Hz to 0Hz.
zeroes2 = np.zeros(len(S21))
S21 = np.append(S21, zeroes2)   # Weird order, but fftfreq requires it

h_t = np.fft.ifft(S21)   # ifft of S21 (time domain)


#


#                       # Initializations

fs = 2*(330 * 1e9)  # Sampling freq (twice the maximum frequency).
ts = 1 / fs     # sampling period
fb = 2*(1e9)   # frequency of bassband signal
tb = 1/fb   # length of each symbol. Note, bassband is not sinusoidal.
# I.e. how fast we are transimitting data. 2GHz == 2Gbit. It can go up to 30Gbit.

symbol_len = 1/(fb * ts)  # length of each symbol in number of bits
symbol_count = 100  # number of symbols we want in our bass band signal

# Time Domain Signal inputs
t0 = 0      # initial time or the start time.
s_z = ts      # step size
s_c = 33000   # step count


#


#                       # Bassband Signal

# below for-loop creates an array of random bits in contiguous blocks.
# e.g. [0, 0, 0, ... 0, 0, 1, 1, 1, ..., 1, 1, 0, 0, 0, ..., 0, 0, ...]
bassband = np.array([])    # Initializaing bitstream to empty numpy array.
for object in range(symbol_count):   # Repeat "symbol_count" number of times

    # rand_b stands for random bit.
    rand_b = (random.randint(0, 3))/3   # PAM 4
    # rand_b = (random.randint(0, 7))/7   # PAM 8

    # random bit is repeated "symbol_len" times and concatenated onto bitstream
    bassband = np.append(bassband, (np.repeat(rand_b, symbol_len)))


#


#                       # Carrier Signal

# "s_c" number of elements each between "t0" and "s_c", spaced "s_z" apart.
time = np.arange(0, s_c) * s_z + t0
carrier_fs = 225 * 1e9  # freq of the waveguide, goes from 220GHz to 330GHz
carrier = np.exp(1j*2*carrier_fs*(np.pi)*time)


#


#                       # Mixing for Input, then Convolving for Output

x_input = carrier * bassband    # Element-wise multiplication, lengths must equal
mag_x = abs(x_input)    # input magnitude

# x_input and h_t, which was the ifft of S21, is convolved to make y_output
y_output = np.convolve(x_input, h_t, mode='full')
mag_y = abs(y_output)   # output magntitude


#


#                       # Plotting I/O in time domain

# Creating x axis and y axis for time domain plots
tx = np.linspace(0, ts*(len(mag_x)), len(mag_x))   # 2000 elements here
ty = np.linspace(0, ts*(len(mag_y)), len(mag_y))   # 2000 elements here

plt.plot(ty, mag_y)     # spikey one
plt.plot(tx, mag_x)     # boxy one
plt.legend(["output_y", "input_x"])
plt.title('Input and Output Signals in the Time Domain')
plt.show()


#                       # Plotting fft of each input signals

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


#


#                       # Time Domain (Eye diagram)

fig, ax = plt.subplots(1, 2)    # This is also used for the frequency plot
marker_size = 1  # how big the markers are on the graph.
eye_diag_window = 3*tb  # arbitrary, we typically display 3 tb in eye diagram
th = np.linspace(0, ts*(len(h_t)), len(h_t))   # 2000 elements here

ax[0].scatter(np.mod(tx, eye_diag_window), mag_x, s=marker_size)
ax[0].scatter(np.mod(ty, eye_diag_window), mag_y, s=marker_size)
ax[0].scatter(np.mod(th, eye_diag_window), abs(h_t), s=marker_size)

ax[0].set_title('Time Domain')
ax[0].legend(["Input Signal", "Output Signal", "ifft($S_{21}$)"])
ax[0].set_ylabel("Amplitude (a.u.)")
ax[0].set_xlabel("Seconds (s)")


#                       # Frequency Domain

# Using fftfreq() to plot S21 and carrier on the correct frequency axis. "s_freq" means sample frequencies
s_freq_S21 = np.fft.fftfreq(len(S21), d=(1/(2*max(freq))))
s_freq_carrier_fft = np.fft.fftfreq(len(carrier_fft), d=(1/(2*max(freq))))

ax[1].plot(s_freq_carrier_fft, abs(carrier_fft)/(np.max(abs(carrier_fft))))
ax[1].plot(s_freq_S21, abs(S21))

ax[1].legend(["275 GHz Carrier", "$S_{21}$"])
ax[1].set_title('Frequency Domain')
ax[1].set_xlabel("Frequency (GHz)")
plt.show()
