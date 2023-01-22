import numpy as np
import geojson
import xarray
import firedrake
from firedrake import assemble, Constant, dx
import icepack


# Load the computed fluidity and velocity fields
# TODO: stop hard-coding this
αs = [2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0]
θs = []
us = []

with firedrake.CheckpointFile("larsen.h5", "r") as chk:
    mesh = chk.load_mesh()
    for α in αs:
        θs.append(chk.load_function(mesh, f"log_fluidity-{α}"))
        us.append(chk.load_function(mesh, f"velocity-{α}"))

    h5f = chk.h5pyfile
    all_indices = np.array(h5f["all_indices"])
    training_indices = np.array(h5f["training_indices"])


# The test indices are the complement of the training indices
all_index_set = set(map(tuple, all_indices))
training_index_set = set(map(tuple, training_indices))
test_indices = np.array(list(all_index_set.difference(training_index_set)))
N = len(test_indices)


# Fetch some velocity data (ask my why it's 6GB)
outline_filename = icepack.datasets.fetch_outline("larsen-2015")
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)
coords = np.array(list(geojson.utils.coords(outline)))
delta = 10e3
xmin, xmax = coords[:, 0].min() - delta, coords[:, 0].max() + delta
ymin, ymax = coords[:, 1].min() - delta, coords[:, 1].max() + delta

velocity_filename = icepack.datasets.fetch_measures_antarctica()
kw = {"x": slice(xmin, xmax), "y": slice(ymax, ymin)}
dataset = xarray.open_dataset(velocity_filename).sel(**kw)
X, Y = dataset["x"], dataset["y"]
xs = np.array([(X[j], Y[i]) for i, j in test_indices])
point_set = firedrake.VertexOnlyMesh(mesh, xs)

V_obs = firedrake.VectorFunctionSpace(point_set, "DG", 0, dim=2)
Q_obs = firedrake.FunctionSpace(point_set, "DG", 0)

u_obs = firedrake.Function(V_obs)
σ_x = firedrake.Function(Q_obs)
σ_y = firedrake.Function(Q_obs)

vx, vy = dataset["VX"], dataset["VY"]
u_obs.dat.data[:] = np.array([(float(vx[i, j]), float(vy[i, j])) for i, j in test_indices])
σ_x.dat.data[:] = np.array([float(dataset["ERRX"][i, j]) for i, j in test_indices])
σ_y.dat.data[:] = np.array([float(dataset["ERRY"][i, j]) for i, j in test_indices])


def loss_functional(u):
    δu = firedrake.interpolate(u, V_obs) - u_obs
    return 0.5 / Constant(N) * ((δu[0] / σ_x)**2 + (δu[1] / σ_y)**2) * dx


for u in us:
    print(assemble(loss_functional(u)))
