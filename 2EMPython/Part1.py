import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# This program goes through the

#                                   Part 1
# Define Important constants
eps_naught = 8.85e-12  # Free space permittivity
K = 1/(4*np.pi*eps_naught)  # Coulomb's constant

# Input the number, position and charge of charges

# MATLAB code
# for n = 1:Nc
#     fprintf('*********************\n Details of charge #%g \n',n);
#     Q_in = input('Enter the position as [x y]:');
#     x(n) = Q_in(1); % Charge position in x
#     y(n) = Q_in(2); % Charge position in y
#     q(n) = input('Enter the charge:'); % Charge of charge
# end

# Python implementation of MATLAB code above
# Nc = int(input("Please enter the number of charges: "))  # number of charges
# for n in range(1, Nc+1):
#     print("*********************\n Details of charge #", n, "\n")
#     Q_in = int(input("Enter the position as [x y]:"))
#     # newArray = numpy.append (a, [10, 11, 12])
# print(Q_in)

# okay, user input will be done last. I will just initialize them myself for now.


# Initializing, 2 charges. Charge 1 at (0,0) with 1C. Charge 2 at (1,1) with -1C.

q = np.array([1, -1, 1])
x = np.array([0, 1, 1])
y = np.array([0, 1, 0])
Nc = 3

print(q, x, y, Nc)  # Just to check


#                                   Part 2
# preallocate storage for the forces on each particle
CFx_t = np.array(np.zeros(Nc, dtype='int64'))
CFy_t = np.array(np.zeros(Nc, dtype='int64'))
print(CFx_t)
print(CFy_t)


# Compute and sum up the coulomb forces on each charge due to other charges
for n in range(Nc):

    # Initialize the total Coulomb forces to zero
    CFx = 0  # Total coulomb forces x - component
    CFy = 0  # Total coulomb forces y - component

    # Compute the force on this charge due to other charges individually
    for m in range(Nc):

        if (n != m):  # Avoid the same charge

            # Compute the distance and its x - and y - components between two charges

            xij = x[n] - x[m]   # Distance x - component
            yij = y[n] - y[m]   # Distance y - component

            # my own note: technically xij and yij should be inputted into the
            # absolute function to ensure they are positive as distance values.
            # It's not done in this code because it's squared in the next line so
            # they didn't bother.

            Rij = np.sqrt(xij*xij + yij*yij)    # Distance magnitude

            # Compute the force between two charges using Coulomb's law
            CFx = CFx + K*q[n]*q[m]*xij/(pow(Rij, 3))  # Force x - component
            CFy = CFy + K*q[n]*q[m]*yij/(pow(Rij, 3))  # Force y - component

    # Sum the forces on this charge due to other charges
    CFx_t[n] = CFx
    CFy_t[n] = CFy

print(CFx_t)
print(CFy_t)


#                                   Part 3
# Plot the charges with total coulomb force

fig, axs = plt.subplots()  # Create a plot.

# This is the define the plot dimenstions - add margins
mar = 2  # The margin to add to the plot
x_min = np.min(x)-mar  # x lower limit
y_min = np.min(y)-mar  # y lower limit
x_max = np.max(x)+mar  # y upper limit
y_max = np.max(y)+mar  # x upper limit
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
print(x_min, x_max, y_min, y_max)


axs.plot(x, y, 'ko')  # Plot the charges locations.

# This bit of code plots the charges' signs.
for m in range(Nc):
    if q[m] > 0:
        axs.text(x[m], y[m], print(m))  # Mark charge
        tempy = y[m] - 0.2
        axs.scatter(x[m], tempy, s=80, c="r", marker="+")  # Mark sign
    if q[m] < 0:
        axs.text(x[m], y[m], print(m))  # Mark charge
        tempy = y[m] - 0.2
        axs.scatter(x[m], tempy, s=50, c="b", marker="_")  # Mark sign

# Displays force vectors as arrows with components(u, v) at the points(x, y).
axs.quiver(x, y, CFx_t, CFy_t)

# Adjust aspect ratio, you need to do this so the plot isn't stretched
plt.gca().set_aspect('equal', adjustable='box')

# Title and axis labelling
plt.title('Coulomb Forces on Arbitrary Charges')
plt.xlabel('Fosition --- X')
plt.ylabel('Position --- Y')


plt.show()
