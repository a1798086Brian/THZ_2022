# pylint: disable=C0103

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import random

# following line enables me to print a numpy array over 1000 elements in its entirety.
np.set_printoptions(threshold=np.inf)


#


#                                               # CODE INFORMATION
#   This code allows the user to plot the real and imaginary part of a sinusoid and compare it to its
#   negative frequency counter part.


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


time = np.arange(0, s_c) * s_z + t0  # 3000 long atm

sinusoid_fs = 500 * 1e5    # 225 kHz
sinusoid = np.exp(1j*2*sinusoid_fs*(np.pi)*time)


#


#


fig, ax = plt.subplots(1, 2)


#                                         # positive frequency
ax[0].plot(time, np.real(sinusoid))
ax[0].plot(time, np.imag(sinusoid))
ax[0].set_title('Positive frequency')
ax[0].legend(["real part", "imaginary part"])
# The absolute would always be 1 so no point plotting that.

ax[0].set_ylabel("Amplitude (a.u.)")
ax[0].set_xlabel("Seconds (s)")


#                                         # negative frequency
sinusoid_fs = -sinusoid_fs    # Inverts frequency
sinusoid = np.exp(1j*2*sinusoid_fs*(np.pi)*time)  # Redefines sinusoid
ax[1].plot(time, np.real(sinusoid))
ax[1].plot(time, np.imag(sinusoid))
ax[1].set_title('Negative frequency')
ax[1].legend(["real part", "imaginary part"])
# The absolute would always be 1 so no point plotting that.

ax[1].set_ylabel("Amplitude (a.u.)")
ax[1].set_xlabel("Seconds (s)")

plt.show()
