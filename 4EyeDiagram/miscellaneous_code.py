

# This line dds 2000 zeroes before S21. The 2000 zeroes means there were no S-parameters for 0-220 Hz. In reality there is,
# ... we just can't picked it up. Why 2000 zeroes? See appendix for further explanation.
zeroes = np.zeros(2000)
# 0s first, then S21. 0-220Hz, then 220-330Hz
S21 = np.append(zeroes, S21)

# Now add 3001 or 3000 (both are probably fine) to the end of S21. This is represents -330Hz to 0Hz.
zeroes2 = np.zeros(len(S21))
S21 = np.append(S21, zeroes2)   # Weird order, but fftfreq requires it


# WHY 2000 ZEROS?
# How I got 2000 is weird. It's because 220GHz to 330Hz requires 1001 numbers. 1001 not 1000 because you have to add an extra...
# ... number when you include both ends. E.g. 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 is actually 11 numbers. Thus, if we...
# ... went from 0 to 220, including 0 and 220, then we would have 221 numbers.


#


#


#


# work out what is the increment in each freq jump. "freq_s_z" stands for frequency step size
freq_s_z = (np.max(freq) - np.min(freq)) / (len(freq) - 1)
# ^^^^^^^^ I didn't use this above code for some reason... It was necessary
#


#


#


# creating a randomly generated input signal. 1000 samples between 0 and 100.
rng = np.random.default_rng(12345)
x = rng.integers(low=0, high=100, size=1000)


#


#


# This bit of code is randomly generating 20 complex numbers as S-parameters
S_param = (rng.integers(low=0, high=100, size=20))/100 + \
    (rng.integers(low=0, high=100, size=20))*1j/100
print(S_param)


# Remember, 1e2 is 100 and 10e2 is 1000


#


#


#                                   # Subplot testing

# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ax = plt.subplots(1, 2)
# fig.suptitle('Horizontally stacked subplots')
# ax1.scatter([1, 2, 3, 4], [1, 2, 3, 4])
# ax1.scatter([1, 2, 3, 4], [-1, -2, -3, -4])
# ax2.scatter([1, 2, 3, 4], [-1, -2, -3, -4])


#


# Remember Sparam is already in freq domain
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


#


#


#       Other miscellaneous print statements that were used as testing statements

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
