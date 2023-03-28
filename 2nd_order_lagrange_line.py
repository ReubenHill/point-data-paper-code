# Plot the 3rd order lagrange nodal basis functions

import atexit
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

x = np.array([0, 0.5, 1])
y0 = np.array([1, 0, 0])
poly0 = lagrange(x, y0)

y1 = np.array([0, 1, 0])
poly1 = lagrange(x, y1)

y2 = np.array([0, 0, 1])
poly2 = lagrange(x, y2)

x_new = np.arange(0, 1.05, 0.05)
fig, ax = plt.subplots(1)
phi0 = Polynomial(poly0.coef[::-1])(x_new)
phi1 = Polynomial(poly1.coef[::-1])(x_new)
phi2 = Polynomial(poly2.coef[::-1])(x_new)
# extend x_new to 0.0 and 1.0 to force the plot to go to 0
x_new = np.append(x_new, [1.0])
x_new = np.append([0.0], x_new)
phi0 = np.append(phi0, [0])
phi0 = np.append([0], phi0)
phi1 = np.append(phi1, [0])
phi1 = np.append([0], phi1)
phi2 = np.append(phi2, [0])
phi2 = np.append([0], phi2)
# extend zeros to the left and right of the plot
x_new = np.append(x_new, [2.0])
x_new = np.append([-2.0], x_new)
phi0 = np.append(phi0, [0])
phi0 = np.append([0], phi0)
phi1 = np.append(phi1, [0])
phi1 = np.append([0], phi1)
phi2 = np.append(phi2, [0])
phi2 = np.append([0], phi2)
ax.plot(x_new, phi0, label="$\phi_0$", linestyle="-")
ax.plot(x_new, phi1, label="$\phi_1$", linestyle="--")
ax.plot(x_new, phi2, label="$\phi_2$", linestyle=":")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 1])
ax.set_xlim([-0.05, 1.05])
ax.set_xlabel("X")
ax.set_ylabel("$\phi_i(X)$")

ax.grid()
ax.legend()
# plt.show()
ax.set_position([0, 0, 1, 1], which="both")
plt.savefig("2nd_order_lagrange_line.pdf", transparent=True, bbox_inches="tight")
