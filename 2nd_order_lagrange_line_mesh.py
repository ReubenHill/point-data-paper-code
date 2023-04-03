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

x_new = np.arange(0, 2.05, 0.05)
fig, ax = plt.subplots(1)
phi0 = Polynomial(poly0.coef[::-1])(x_new)
phi0[x_new > 1.0] = 0
phi1 = Polynomial(poly1.coef[::-1])(x_new)
phi1[x_new > 1.0] = 0
phi2 = Polynomial(poly2.coef[::-1])(x_new)
phi2[x_new >= 1.0] = phi0[x_new <= 1.0]
# phi3 is phi1 shifted to the right by 1
phi3 = np.zeros_like(phi2)
phi3[x_new >= 1.0] = phi1[x_new <= 1.0]
# phi4 is phi2 shifted to the right by 1
phi4 = np.zeros_like(phi0)
phi4[x_new >= 1.0] = phi2[x_new <= 1.0]

# from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
longdashed = (5, (10, 3))
densely_dashdotted = (0, (3, 1, 1, 1))

ax.plot(x_new, phi0, label="$\phi_0$", linestyle="-")
ax.plot(x_new, phi1, label="$\phi_1$", linestyle="--")
ax.plot(x_new, phi2, label="$\phi_2$", linestyle=":")
ax.plot(x_new, phi3, label="$\phi_3$", linestyle="-.")
ax.plot(x_new, phi4, label="$\phi_4$", linestyle=densely_dashdotted)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1])
ax.set_xlim([-0.0, 2.0])
ax.set_xlabel("x")
ax.set_ylabel("$\phi_i(x)$")
ax.grid(axis="x")
ax.legend(loc="center", bbox_to_anchor=(0.75, 0.55))

# plt.show()
ax.set_position([0, 0, 1, 1], which="both")
fig.set_size_inches(9, 4)
plt.savefig("2nd_order_lagrange_line_mesh.pdf", transparent=True, bbox_inches="tight")
