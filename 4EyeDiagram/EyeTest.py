# pylint: disable=C0103

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import random


# following line enables me to print a numpy array over 1000 elements in its entirety.
np.set_printoptions(threshold=np.inf)


#


#                                               # Eye Diagram Simulation
t = np.linspace(0, 100, 2000)
x = np.sin(t)

# f=np.fft.fftfreq(spad.size,d=ts) # spad = bassband.sz, ts= sampling freq
# # Harry used this to plot right side diagram.

#                                               # Wrap Arround

# We need 3 symbols long. How long is one symbol? That depends on our bitstream.
# Symbol length here is not the same as symbol length in previous code. That was for symbol length in bits. Here is in time.

fb = 2*(1e9)    # fb is the frequency of the bassband signal in Hz
tb = 1/fb   # tb is length of each symbol. Note, bassband is not sinusoidal.

eye_diag_window = 3*tb  # arbitrary, we typically display 3 tb in eye diagram


#


# Testing scatter
plt.scatter(np.mod(range(1, 11), 3), range(
    1, 11), s=1, alpha=0.8, marker='D')
plt.show()


#


# Testing np.mod()
print(np.mod(range(1, 11), 3))


#


#


#


# Do I use fft.fftfreq here?

# WRAP IN TIME DOMAIN!!!


# wrap_freq = 2;  #Hz
# wrap_period = 1/wrap_freq

# Plan is to divite into multiple smaller sub-signals by storing them in arrays and then plot it.


#                                               # "Colour Temperature / Heat Map"


# no idea for this part, not necessary is it?


#                                               # Plot
# plt.plot(t, x)
# plt.show()
