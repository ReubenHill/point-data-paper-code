import json
import argparse
import numpy as np
from numpy import pi as π
import firedrake
from firedrake import (
    interpolate, assemble, conditional, exp, Constant, inner, grad, dx, ds
)
from firedrake_adjoint import ReducedFunctional, Control, minimize

parser = argparse.ArgumentParser()
parser.add_argument("--num-observation-wells", type=int, default=6)
parser.add_argument("--num-observation-times", type=int, default=3)
parser.add_argument("--num-trials", type=int, default=30)
parser.add_argument("--std-dev", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=1729)
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

# Read in the exact solution
with firedrake.CheckpointFile(args.input, "r") as chk:
    timestep = chk.get_attr("/", "timestep")
    num_steps = chk.get_attr("/", "num-steps")

    mesh = chk.load_mesh()
    φs_exact = [
        chk.load_function(mesh, name="hydraulic-head", idx=idx)
        for idx in range(num_steps + 1)
    ]

Lx, Ly = 6500.0, 4500.0

x = firedrake.SpatialCoordinate(mesh)

L_1 = Constant(2500.0)
L_2 = Constant(4500.0)

S = Constant(0.0001)
T_1 = Constant(1000.0)
T_2 = Constant(1000.0)
T_3 = Constant(1000.0)
T = conditional(
    x[0] < L_1,
    T_1,
    conditional(
        x[0] < L_2,
        T_2,
        T_3,
    )
)


# Create the synthetic observations
N = args.num_observation_wells + 1
Y = np.array([Ly * i / N for i in range(1, N)])
X1 = float(L_1) / 2 * np.ones(N - 1)
X2 = (float(L_1) + float(L_2)) / 2 * np.ones(N - 1)
X3 = (float(L_2) + Lx) / 2 * np.ones(N - 1)
xs1 = np.column_stack((X1, Y))
xs2 = np.column_stack((X2, Y))
xs3 = np.column_stack((X3, Y))
xs = np.vstack((xs1, xs2, xs3))

observation_wells = firedrake.VertexOnlyMesh(mesh, xs)
Z = firedrake.FunctionSpace(observation_wells, "DG", 0)

pumping_rate = Constant(2000.0)
a = Constant(0.75)
z = Constant((a * Lx + (1 - a) * L_2, Ly / 2))

r = Constant(300.0)
area = π * r**2
expr = conditional(inner(x - z, x - z) < r**2, pumping_rate / area, 0.0)
Q = firedrake.FunctionSpace(mesh, "CG", 1)
q = interpolate(expr, Q)

V = φs_exact[0].function_space()
φ_0 = Constant(100.0)
φ = interpolate(φ_0, V)

dirichlet_ids = [1]
bc = firedrake.DirichletBC(V, φ_0, dirichlet_ids)

final_time = num_steps * timestep
times = np.linspace(0.0, final_time, num_steps + 1)
obs_times = np.linspace(0.0, 1.0, args.num_observation_times + 1)[1:] * final_time
obs_indices = [abs(times - t).argmin() for t in obs_times]

results = {
    "num-observation-wells": args.num_observation_wells,
    "num-observation-times": args.num_observation_times,
    "std-dev": args.std_dev,
    "transmissivities": [],
    "mean-square-errors": [],
}

pcg = firedrake.PCG64(seed=args.seed)
rng = firedrake.RandomGenerator(pcg)

for trial in range(args.num_trials):
    φ = interpolate(φ_0, V)
    dt = Constant(timestep)

    ψ = firedrake.TestFunction(V)
    φ_n = φ.copy(deepcopy=True)

    F = (S * (φ - φ_n) / dt * ψ + T * inner(grad(φ), grad(ψ)) + q * ψ) * dx
    problem = firedrake.NonlinearVariationalProblem(F, φ, bc)
    solver = firedrake.NonlinearVariationalSolver(problem)

    φs = [φ.copy(deepcopy=True)]
    for step in range(num_steps):
        solver.solve()
        φ_n.assign(φ)
        φs.append(φ.copy(deepcopy=True))

    observed_heads = [
        interpolate(φs_exact[index], Z) + rng.normal(Z, 0.0, args.std_dev)
        for index in obs_indices
    ]
    computed_heads = [interpolate(φs[index], Z) for index in obs_indices]

    N = Constant(len(xs))
    σ = Constant(args.std_dev)
    squared_errors = sum(
        assemble(0.5 / N * (φ - φ_obs)**2 / σ**2 * dx)
        for φ, φ_obs in zip(computed_heads, observed_heads)
    )

    J = ReducedFunctional(squared_errors, [Control(T) for T in [T_1, T_2, T_3]])
    T_opt = minimize(J)
    results["transmissivities"].append([float(T) for T in T_opt])
    results["mean-square-errors"].append(float(J(T_opt)))

with open(args.output, "w") as output_file:
    json.dump(results, output_file, indent=2)
