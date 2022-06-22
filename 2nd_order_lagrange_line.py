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
ax.plot(x_new, Polynomial(poly0.coef[::-1])(x_new), label='$\phi_0$')
ax.plot(x_new, Polynomial(poly1.coef[::-1])(x_new), label='$\phi_1$')
ax.plot(x_new, Polynomial(poly2.coef[::-1])(x_new), label='$\phi_2$')
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 1])

ax.grid()
ax.legend()
# plt.show()
ax.set_position([0, 0, 1, 1], which='both')
plt.savefig('2nd_order_lagrange_line.svg', transparent = True, bbox_inches='tight')