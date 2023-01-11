import argparse
import subprocess
import numpy as np
import geojson
import xarray
import firedrake
import icepack


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="larsen.h5")
parser.add_argument("--random-seed", type=int, default=1729)
parser.add_argument("--data-fraction", type=float, default=0.05)
args = parser.parse_args()


# Fetch an outline of the ice shelf and make a mesh
outline_filename = icepack.datasets.fetch_outline("larsen-2015")
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry = icepack.meshing.collection_to_geo(outline)
with open("larsen.geo", "w") as geo_file:
    geo_file.write(geometry.get_code())

command = "gmsh -2 -format msh2 -v 2 -o larsen.msh larsen.geo"
subprocess.run(command.split())
mesh = firedrake.Mesh("larsen.msh")

coords = np.array(list(geojson.utils.coords(outline)))
delta = 10e3
xmin, xmax = coords[:, 0].min() - delta, coords[:, 0].max() + delta
ymin, ymax = coords[:, 1].min() - delta, coords[:, 1].max() + delta


# Fetch some velocity data
velocity_filename = icepack.datasets.fetch_measures_antarctica()
kw = {"x": slice(xmin, xmax), "y": slice(ymax, ymin)}
dataset = xarray.open_dataset(velocity_filename).sel(**kw)


# Find the indices of all grid points that are inside the mesh
X, Y = dataset["x"], dataset["y"]
all_indices = np.array(
    [
        (i, j)
        for i, y in enumerate(Y) for j, x in enumerate(X)
        if mesh.locate_cell((x, y))
    ]
)


# Randomly select a fraction of the indices to use for training
rng = np.random.default_rng(seed=args.random_seed)
N = int(args.data_fraction * len(all_indices))
training_indices = rng.choice(all_indices, size=N, axis=0)


# Write the results to disk
with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_mesh(mesh)
    f = chk.h5pyfile
    f.create_dataset("all_indices", data=all_indices)
    f.create_dataset("training_indices", data=training_indices)
