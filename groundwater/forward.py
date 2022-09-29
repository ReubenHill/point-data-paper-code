import argparse
import numpy as np
from numpy import pi as π
import firedrake
from firedrake import conditional, exp, Constant, inner, grad, dx, ds

parser = argparse.ArgumentParser()
parser.add_argument("--timestep", type=float, default=1.0 / 24)
parser.add_argument("--output")
parser.add_argument("--num-points", type=int, default=65)
parser.add_argument("--degree", type=int, default=1)
args = parser.parse_args()

# The following problem is taken from problem 4.2.1 in Ne-Zheng Sun,
# "Inverse Problems in Groundwater Modeling."

Lx, Ly = 6500.0, 4500.0
nx = args.num_points
ny = int((Ly * nx) / Lx)
mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)

x = firedrake.SpatialCoordinate(mesh)

# Coordinates of each zone of the aquifer
L_1 = Constant(2500.0)
L_2 = Constant(4500.0)

# Transmissivity of the aquifer (m^2 / day)
T_1 = Constant(500)
T_2 = Constant(1000.0)
T_3 = Constant(2000.0)

T = conditional(
    x[0] < L_1,
    T_1,
    conditional(
        x[0] < L_2,
        T_2,
        T_3,
    )
)

# Storativity of the aquifer
S = Constant(0.0001)

# Pumping rate (m^3 / day) and location of pumping well, 3/4 of the way from
# the boundary of the third zone to the right-hand side of the domain
pumping_rate = Constant(2000.0)
a = Constant(0.75)
z = Constant((a * Lx + (1 - a) * L_2, Ly / 2))

# TODO: Make it a genuine point source
"""
xs = np.array([(float(z[0]), float(z[1]))])
pumping_wells = firedrake.VertexOnlyMesh(mesh, xs)
Z = firedrake.FunctionSpace(pumping_wells, "DG", 0)
"""

r = Constant(300.0)
area = π * r**2
expr = conditional(inner(x - z, x - z) < r**2, pumping_rate / area, 0.0)
Q = firedrake.FunctionSpace(mesh, "CG", 1)
q = firedrake.interpolate(expr, Q)


# Initial value of the hydraulic head is 100m everywhere
V = firedrake.FunctionSpace(mesh, "CG", args.degree)
φ_0 = Constant(100.0)
φ = firedrake.interpolate(φ_0, V)

# Dirichlet boundary conditions -- the pressure head is a constant 100m at the
# left-hand boundary
dirichlet_ids = [1]
bc = firedrake.DirichletBC(V, φ_0, dirichlet_ids)

ψ = firedrake.TestFunction(V)
φ_n = φ.copy(deepcopy=True)

dt = Constant(args.timestep)
F = (S * (φ - φ_n) / dt * ψ + T * inner(grad(φ), grad(ψ)) + q * ψ) * dx
problem = firedrake.NonlinearVariationalProblem(F, φ, bc)
solver = firedrake.NonlinearVariationalSolver(problem)

final_time = 1.5
num_steps = int(final_time / float(dt))

with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.set_attr("/", "timestep", float(dt))
    chk.set_attr("/", "num-steps", num_steps)

    chk.save_function(φ, name="hydraulic-head", idx=0)
    for step in range(num_steps):
        solver.solve()
        φ_n.assign(φ)
        chk.save_function(φ, name="hydraulic-head", idx=step + 1)
