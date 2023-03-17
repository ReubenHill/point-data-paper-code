import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import firedrake
from firedrake import Constant, conditional

parser = argparse.ArgumentParser()
parser.add_argument("--input")
args = parser.parse_args()

with firedrake.CheckpointFile(args.input, "r") as chk:
    timestep = chk.get_attr("/", "timestep")
    num_steps = chk.get_attr("/", "num-steps")

    mesh = chk.load_mesh()
    φs = [
        chk.load_function(mesh, name="hydraulic-head", idx=idx)
        for idx in range(num_steps + 1)
    ]

fig, ax = plt.subplots()
ax.set_title("Aquifer at 1.5 days simulation time")
ax.set_aspect("equal")
ax.set_xlabel("easting (m)")
ax.set_ylabel("northing")
colors = firedrake.tripcolor(φs[-1], axes=ax)

Lx, Ly = 6500.0, 4500.0
L_1 = 2500.0
L_2 = 4500.0

for num_observation_wells, marker, color in zip([3, 6], ["^", "*"], ["tab:red", "k"]):
    N = num_observation_wells + 1
    Y = np.array([Ly * i / N for i in range(1, N)])
    X1 = L_1 / 2 * np.ones(N - 1)
    X2 = (L_1 + L_2) / 2 * np.ones(N - 1)
    X3 = (L_2 + Lx) / 2 * np.ones(N - 1)
    xs1 = np.column_stack((X1, Y))
    xs2 = np.column_stack((X2, Y))
    xs3 = np.column_stack((X3, Y))
    xs = np.vstack((xs1, xs2, xs3))
    label = f"num wells = {num_observation_wells}"
    ax.scatter(xs[:, 0], xs[:, 1], 8, marker=marker, label=label, color=color)

ax.legend(bbox_to_anchor=(-0.1, -0.1), loc="upper left", borderaxespad=0)
fig.colorbar(colors, label="hydraulic head (m)")
fig.savefig("hydraulic-head-final.pdf")


Lx, Ly = 6500.0, 4500.0
x = firedrake.SpatialCoordinate(mesh)
L_1 = Constant(2500.0)
L_2 = Constant(4500.0)
S = Constant(0.0001)
T_1 = Constant(500)
T_2 = Constant(1000.0)
T_3 = Constant(2000.0)
expr = conditional(
    x[0] < L_1,
    T_1,
    conditional(
        x[0] < L_2,
        T_2,
        T_3,
    )
)
Q = firedrake.FunctionSpace(mesh, "DG", 0)
T = firedrake.project(expr, Q)

fig, ax = plt.subplots()
ax.set_title("Exact aquifer transmissivity")
ax.set_aspect("equal")
ax.set_xlabel("easting (m)")
ax.set_ylabel("northing")
colors = firedrake.tripcolor(T, axes=ax)
fig.colorbar(colors, label="m${}^2$ / day")
fig.savefig("transmissivity-exact.pdf")


experiments = []
for filename in [f"experiment{index}.json" for index in [2, 4]]:
    with open(filename, "r") as input_file:
        experiments.append(json.load(input_file))

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.set_title("Inverse problem solution spread")
ax.set_xlabel("transmissivity (m${}^2$/day)")
ax.set_ylabel("probability density")
colors = ["tab:blue", "tab:orange"]
labels = ["6 wells, 3 times", "3 wells, 6 times"]
for experiment, color, label in zip(experiments, colors, labels):
    for index in range(3):
        data = np.array(experiment["transmissivities"])
        mean = data[:, index].mean()
        std = data[:, index].std()
        tmin, tmax = mean - 10 * std, mean + 10 * std
        ts = np.linspace(tmin, tmax, 200)
        ps = np.exp(-(ts - mean)**2 / (2 * std**2)) / np.sqrt(2 * np.pi * std**2)
        ax.plot(ts, ps, color=color, label=label)

# Thanks to https://stackoverflow.com/a/13589144/
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

fig.savefig("transmissivity-probability-densities.pdf", bbox_inches="tight")
