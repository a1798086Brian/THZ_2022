# pylint: disable=C0103
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random

# following line enables me to print a numpy array over 1000 elements in its entirety.
np.set_printoptions(threshold=np.inf)
#
#
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


def eye(freq, Re, Im):

    #                       # S-parameters and h_t

    # These are transfer functions. h_t is like h(t). S21 is like H(w). Both h_t and S21 are 6002 long.
    S21 = Re + 1j*Im     # S21 is basically the Scattering parameters

    # We need to do funky business to S21 later, but we want to save a copy of the original S21 now, because that is...
    # The version we want to return later.
    S21_output = S21

    # Now we will tinker with out S21 values.
    # This line dds 2000 zeroes before S21. The 2000 zeroes means there were no S-parameters for 0-220 Hz. In reality there is,
    # ... we just can't picked it up. Why 2000 zeroes? See appendix for further explanation.
    zeroes = np.zeros(2000)
    # 0s first, then S21. 0-220Hz, then 220-330Hz
    S21 = np.append(zeroes, S21)

    # Now add 3001 or 3000 (both are probably fine) to the end of S21. This is represents -330Hz to 0Hz.
    zeroes2 = np.zeros(len(S21))
    S21 = np.append(S21, zeroes2)   # Weird order, but fftfreq requires it

    h_t = np.fft.ifft(S21)   # ifft of S21 (time domain)
    #
    #
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
    #
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
    #
    #
    #                       # Carrier Signal

    # "s_c" number of elements each between "t0" and "s_c", spaced "s_z" apart.
    time = np.arange(0, s_c) * s_z + t0
    carrier_fs = 225 * 1e9  # freq of the waveguide, goes from 220GHz to 330GHz
    carrier = np.exp(1j*2*carrier_fs*(np.pi)*time)
    #
    #
    #
    #                       # Mixing for Input, then Convolving for Output

    x_input = carrier * bassband    # Element-wise multiplication, lengths must equal
    mag_x = abs(x_input)    # input magnitude

    # x_input and h_t, which was the ifft of S21, is convolved to make y_output
    y_output = np.convolve(x_input, h_t, mode='full')
    mag_y = abs(y_output)   # output magntitude
    #
    #
    #
    #                       # Final Calculations and Return Values
    ty = np.linspace(0, ts*(len(mag_y)), len(mag_y))
    eye_diag_window = 3*tb  # arbitrary, we typically display 3 tb in eye diagram
    warped_ty = np.mod(ty, eye_diag_window)
    s_freq_S21 = np.fft.fftfreq(len(S21), d=(1/(2*max(freq))))

    abs_S21 = abs(S21)

    s_freq_S21_rearranged = np.concatenate((
        s_freq_S21[3001:6002], s_freq_S21[0:3000]), axis=None)

    abs_S21_rearranged = np.concatenate((
        abs_S21[3001:6002], abs_S21[0:3000]), axis=None)

    return warped_ty, mag_y, s_freq_S21_rearranged, abs_S21_rearranged


a, b, c, d = eye(freq, Re, Im)
print(d[2000:3000])
# x_stream_a = s_freq_S21
# y_stream_a = abs(S21)
# x_stream_c = warped_ty
# y_stream_c = mag_y
