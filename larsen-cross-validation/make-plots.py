import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import xarray
import firedrake
import icepack

# Plot the test errors
with open("test-errors.json", "r") as test_errors_file:
    test_errors = np.array(json.load(test_errors_file))

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.scatter(test_errors[:, 0], test_errors[:, 1])
ax.set_xlabel("Regularization parameter (meters)")
ax.set_ylabel("Mean-square error on test set")
ax.set_title("Cross-validation error")
fig.savefig("xval-test-errors.png", dpi=150)


# Get the simulation data
αs = np.linspace(2750.0, 5000.0, 10)
θs = []

with firedrake.CheckpointFile("larsen.h5", "r") as chk:
    mesh = chk.load_mesh()
    for α in αs:
        θs.append(chk.load_function(mesh, f"log_fluidity-{α}"))

    training_indices = np.array(chk.h5pyfile["training_indices"])

coords = mesh.coordinates.dat.data_ro
δ = 10e3
xmin, xmax = coords[:, 0].min() - δ, coords[:, 0].max() + δ
ymin, ymax = coords[:, 1].min() - δ, coords[:, 1].max() + δ

# Get the training points
velocity_filename = icepack.datasets.fetch_measures_antarctica()
kw = {"x": slice(xmin, xmax), "y": slice(ymax, ymin)}
dataset = xarray.open_dataset(velocity_filename).sel(**kw)
X, Y = dataset["x"], dataset["y"]
xs = np.array([(X[j], Y[i]) for i, j in training_indices])

# Fetch a satellite image
image_filename = icepack.datasets.fetch_mosaic_of_antarctica()
with rasterio.open(image_filename, "r") as image_file:
    transform = image_file.transform
    window = rasterio.windows.from_bounds(
        left=xmin,
        bottom=ymin,
        right=xmax,
        top=ymax,
        transform=transform,
    )
    image = image_file.read(indexes=1, window=window, masked=True)

# Plot some of the log-fluidity values
fig, axes = plt.subplots(
    nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6.4, 3.6)
)
#xmin, ymin, xmax, ymax = rasterio.windows.bounds(window, transform)
kw = {
    "extent": (xmin, xmax, ymin, ymax),
    "cmap": "Greys_r",
    "vmin": 12e3,
    "vmax": 16.38e3,
}
for ax in axes.flatten():
    ax.imshow(image, **kw)
firedrake.tripcolor(θs[2], vmin=-5, vmax=+5, axes=axes[0])
firedrake.tripcolor(θs[-1], vmin=-5, vmax=+5, axes=axes[1])
axes[0].set_xlabel("easting (meters)")
axes[0].set_ylabel("northing (meters)")
axes[1].get_xaxis().set_visible(False)
axes[0].set_title(f"$\\alpha$ = {αs[2] / 1e3}km")
axes[1].set_title(f"$\\alpha$ = {αs[-1] / 1e3}km")
fig.suptitle("Log-fluidities at different regularization")
fig.savefig("xval-log-fluidities.png", dpi=150, bbox_inches="tight")

# Show where the training points are
fig, axes = plt.subplots()
axes.set_aspect("equal")
axes.imshow(image, **kw)
bxmin, bxmax = -2.075e6, -2.025e6
bymin, bymax = 1.125e6, 1.175e6
axes.set_xlim((bxmin, bxmax))
axes.set_ylim((bymin, bymax))
firedrake.triplot(mesh, axes=axes)
axes.scatter(xs[:, 0], xs[:, 1], marker=".")
axes.set_title("Training point locations")
fig.savefig("xval-training-points.png", dpi=150, bbox_inches="tight")

fig, axes = plt.subplots()
axes.set_aspect("equal")
axes.imshow(image, **kw)
axes.plot([bxmin, bxmax, bxmax, bxmin, bxmin], [bymin, bymin, bymax, bymax, bymin])
firedrake.triplot(mesh, interior_kw={"linewidth": 0.0}, axes=axes)
axes.set_title("Larsen C Ice Shelf")
fig.savefig("larsen-c.png")
