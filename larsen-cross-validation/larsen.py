import argparse
import subprocess
import numpy as np
import geojson
import xarray
import firedrake
from firedrake import assemble, Constant, inner, grad, dx
import icepack
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator


parser = argparse.ArgumentParser()
parser.add_argument("--regularization", type=float, default=2.5e3)
parser.add_argument("--random-seed", type=int, default=1729)
parser.add_argument("--data-fraction", type=float, default=0.05)
parser.add_argument("--output", default="larsen.h5")
parser.add_argument("--suffix", default="")
args = parser.parse_args()


# Fetch a GeoJSON file describing the outline of the domain, convert it to
# gmsh's input format, generate a mesh, and make some function spaces.
outline_filename = icepack.datasets.fetch_outline("larsen-2015")
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry = icepack.meshing.collection_to_geo(outline)
with open("larsen.geo", "w") as geo_file:
    geo_file.write(geometry.get_code())

command = "gmsh -2 -format msh2 -v 2 -o larsen.msh larsen.geo"
subprocess.run(command.split())
mesh = firedrake.Mesh("larsen.msh")
Q = firedrake.FunctionSpace(mesh, "CG", 2)
V = firedrake.VectorFunctionSpace(mesh, "CG", 2)


# Load in the thickness map and apply a 2km smoothing filter. The raw thickness
# map resolves individual cracks.
thickness_filename = icepack.datasets.fetch_bedmachine_antarctica()
with xarray.open_dataset(thickness_filename) as dataset:
    h_obs = icepack.interpolate(dataset["thickness"], Q)

h = h_obs.copy(deepcopy=True)
α = Constant(2e3)
J = 0.5 * ((h - h_obs) ** 2 + α**2 * inner(grad(h), grad(h))) * dx
F = firedrake.derivative(J, h)
firedrake.solve(F == 0, h)


# Fetch some velocity data (ask my why it's 6GB).
coords = np.array(list(geojson.utils.coords(outline)))
delta = 10e3
xmin, xmax = coords[:, 0].min() - delta, coords[:, 0].max() + delta
ymin, ymax = coords[:, 1].min() - delta, coords[:, 1].max() + delta

velocity_filename = icepack.datasets.fetch_measures_antarctica()
kw = {"x": slice(xmin, xmax), "y": slice(ymax, ymin)}
dataset = xarray.open_dataset(velocity_filename).sel(**kw)
u_initial = icepack.interpolate((dataset["VX"], dataset["VY"]), V)


# Randomly select a sub-sample of points and create the observational data
X, Y = dataset["x"], dataset["y"]
all_indices = np.array(
    [
        (i, j)
        for i, y in enumerate(Y) for j, x in enumerate(X)
        if mesh.locate_cell((x, y))
    ]
)
rng = np.random.default_rng(seed=args.random_seed)
N = int(args.data_fraction * len(all_indices))
indices = rng.choice(all_indices, size=N, axis=0)

xs = np.array([(X[j], Y[i]) for i, j in indices])
point_set = firedrake.VertexOnlyMesh(mesh, xs)
V_obs = firedrake.VectorFunctionSpace(point_set, "DG", 0, dim=2)
Q_obs = firedrake.FunctionSpace(point_set, "DG", 0)

u_obs = firedrake.Function(V_obs)
σ_x = firedrake.Function(Q_obs)
σ_y = firedrake.Function(Q_obs)

vx, vy = dataset["VX"], dataset["VY"]
u_obs.dat.data[:] = np.array([(float(vx[i, j]), float(vy[i, j])) for i, j in indices])
σ_x.dat.data[:] = np.array([float(dataset["ERRX"][i, j]) for i, j in indices])
σ_y.dat.data[:] = np.array([float(dataset["ERRY"][i, j]) for i, j in indices])


# Define the physics model.
# Here we're re-parameterizing the fluidity $A$ of the ice shelf in terms of
# an auxiliary field $\theta$ according to
#
# $$A = A_0e^\theta$$
#
# where $A_0$ is the ice fluidity at -13C.
T = Constant(260)
A0 = icepack.rate_factor(T)

def viscosity(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    θ = kwargs["log_fluidity"]

    A = A0 * firedrake.exp(θ)
    return icepack.models.viscosity.viscosity_depth_averaged(
        velocity=u, thickness=h, fluidity=A
    )


model = icepack.models.IceShelf(viscosity=viscosity)


# Create an object for solving the momentum conservation equation.
opts = {
    "dirichlet_ids": [2, 4, 5, 6, 7, 8, 9],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_max_it": 100,
        "snes_type": "newtontr",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}
solver = icepack.solvers.FlowSolver(model, **opts)

θ = firedrake.Function(Q)
u = solver.diagnostic_solve(
    velocity=u_initial,
    thickness=h,
    log_fluidity=θ,
)

def simulation(θ):
    return solver.diagnostic_solve(
        velocity=u_initial, thickness=h, log_fluidity=θ
    )


# Define the loss and regularization functional. We're dividing by the number
# of observations and the area of the domain respectively to make everything
# dimensionless.
def loss_functional(u):
    δu = firedrake.interpolate(u, V_obs) - u_obs
    return 0.5 / Constant(N) * ((δu[0] / σ_x)**2 + (δu[1] / σ_y)**2) * dx


area = assemble(Constant(1) * dx(mesh))
Ω = Constant(area)

def regularization(θ):
    L = Constant(args.regularization)
    return 0.5 * L**2 / Ω * inner(grad(θ), grad(θ)) * dx


# Finally, define the inverse problem we wish to solve and create an object to
# solve it.
stats_problem = StatisticsProblem(
    simulation=simulation,
    loss_functional=loss_functional,
    regularization=regularization,
    controls=θ,
)

estimator = MaximumProbabilityEstimator(
    stats_problem,
    gradient_tolerance=1e-6,
    step_tolerance=1e-2,
    max_iterations=150,
)
θ = estimator.solve()
u = simulation(θ)


# Write the results to a file.
with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_mesh(mesh)
    chk.save_function(u, name=f"velocity{args.suffix}")
    chk.save_function(θ, name=f"log_fluidity{args.suffix}")

    f = chk.h5pyfile
    f.create_dataset("all_indices", data=all_indices)
    f.create_dataset("training_indices", data=indices)
