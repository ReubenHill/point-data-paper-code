import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
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


# Plot some values of the inferred log-fluidity
αs = np.linspace(2750.0, 5000.0, 10)
θs = []

with firedrake.CheckpointFile("larsen.h5", "r") as chk:
    mesh = chk.load_mesh()
    for α in αs:
        θs.append(chk.load_function(mesh, f"log_fluidity-{α}"))

coords = mesh.coordinates.dat.data_ro
δ = 50e3
xmin, xmax = coords[:, 0].min() - δ, coords[:, 0].max() + δ
ymin, ymax = coords[:, 1].min() - δ, coords[:, 1].max() + δ

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

fig, axes = plt.subplots(
    nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6.4, 3.6)
)
xmin, ymin, xmax, ymax = rasterio.windows.bounds(window, transform)
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
