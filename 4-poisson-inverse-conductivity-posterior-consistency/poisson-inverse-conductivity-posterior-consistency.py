#!/usr/bin/env python
# coding: utf-8

# # Problem Description
#
# We want to find out how the solution of our inverse problem converges as we increase the number of points for both the new and traditional methods of data interpolation.
#
# If we have what is known as **"posterior consistency"** then we expect that the error in our solution, when compared to the true solution, will always decrease as we increase the number of points we are assimilating.
#
# ## Posterior Consistency **NEEDS WORK**
#
# From a Bayesian point of view, the regularisation we choose and the weighting we give it encode information about our assumed prior probability distribution of $q$ before we start assimilating data (adding observations).
# Take, for example, the regularisation used in the this problem
#
# $$
# \frac{\alpha^2}{2}\int_\Omega|\nabla q|^2dx
# $$
#
# which asserts a prior that the solution $q$ which minimises $J$ should be smooth and gives a weighting $\alpha$ to the assertion.
# If we have posterior consistency, the contribution of increasing numbers of measurements $u_{obs}$ should increase the weighting of our data relative to our prior and we should converge towards the true solution.
#
# ## Hypothesis
#
# Our two methods minimise two different functionals.
# The first minimises $J$
#
# $$J[u, q] =
# \underbrace{\frac{1}{2}\int_{\Omega_v}\left(\frac{u_{obs} - I(u, \text{P0DG}(\Omega_v))}{\sigma}\right)^2dx}_{\text{model-data misfit}} +
# \underbrace{\frac{\alpha^2}{2}\int_\Omega|\nabla q|^2dx}_{\text{regularization}}$$
#
# whilst the second minimises $J'$
#
# $$J'[u, q] =
# \underbrace{\frac{1}{2}\int_{\Omega}\left(\frac{u_{interpolated} - u}{\sigma}\right)^2dx}_{\text{model-data misfit}} +
# \underbrace{\frac{\alpha^2}{2}\int_\Omega|\nabla q|^2dx}_{\text{regularization}}.$$
#
# As set up here increasing the number of points to assimilate has the effect of increasing the size of the misfit term in $J$ (with a weight proportional to each measurement's variance $\sigma$ so we expect to converge to $q_{true}$ as the number of measurements increases.
#
# As we increase the number of measurements in $J'$ we have to hope that (a) our calculated $u_{interpolated}$ approaches $u$ (to minimise the misfit) and (b) we do not expect the misfit term to increase relative to the regularizatuion term since it doesn't get relatively bigger.
#
# I therefore predict that minimising $J$ will display posterior consistency and that minimising the various $J'$ for each $u_{interpolated}$ will not.
# Who knows what we will converge to!
#
# ## Hypothesis Amenendment! A note on finite element method error
# Note that our solutions all exist in finite element spaces which are usually approximations of a true solution with some error that (hopefully) decreases as mesh density increase and solution space order increase.
# Since I am comparing to a solution $u_true$ in CG2 space I expect, at best, that we will converge to $u_true$ when we have, on average, enough points per cell to fully specify the lagrange polynomials in that cell.
# Were we in CG1 this would be 3 points per cell (I can't remember how many we would need for CG2!) to give convergence if those measurements had no noise.
# Since our measurements are noisy I do not expect actual convergence, but I anticipate some slowing in convergence.

# # Setup

# In[1]:
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    CloughTocher2DInterpolator,
    Rbf,
)

import matplotlib.pyplot as plt
import firedrake
import firedrake_adjoint

from firedrake import Constant, cos, sin

import numpy as np
from numpy import pi as π
from numpy import random

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ## Fake $q_{true}$

# In[2]:

mesh = firedrake.UnitSquareMesh(32, 32)

# Solution Space
V = firedrake.FunctionSpace(mesh, family='CG', degree=2)

# q (Control) Space
Q = firedrake.FunctionSpace(mesh, family='CG', degree=2)

seed = 1729
generator = random.default_rng(seed)

degree = 5
x = firedrake.SpatialCoordinate(mesh)

q_true = firedrake.Function(Q)
for k in range(degree):
    for l in range(int(np.sqrt(degree**2 - k**2))):
        Z = np.sqrt(1 + k**2 + l**2)
        ϕ = 2 * π * (k * x[0] + l * x[1])

        A_kl = generator.standard_normal() / Z
        B_kl = generator.standard_normal() / Z

        expr = Constant(A_kl) * cos(ϕ) + Constant(B_kl) * sin(ϕ)
        mode = firedrake.interpolate(expr, Q)

        q_true += mode

print('Made fake q_true')

# ## Fake $u_{true}$

# In[3]:


from firedrake import exp, inner, grad, dx
u_true = firedrake.Function(V)
v = firedrake.TestFunction(V)
f = Constant(1.0)
k0 = Constant(0.5)
bc = firedrake.DirichletBC(V, 0, 'on_boundary')
F = (k0 * exp(q_true) * inner(grad(u_true), grad(v)) - f * v) * dx
firedrake.solve(F == 0, u_true, bc)

print('Made fake u_true')

# Clear tape since don't need to have taped above
tape = firedrake_adjoint.get_working_tape()
tape.clear_tape()

# ## Generating Observational Data $u_{obs}$
# We run up in powers of 2 until we have plenty of observations per cell (on average)

# In[4]:

signal_to_noise = 20
U = u_true.dat.data_ro[:]
u_range = U.max() - U.min()
σ = firedrake.Constant(u_range / signal_to_noise)

methods = ['point-cloud', 'nearest', 'linear', 'clough-tocher', 'gaussian']

min_power_of_2 = 2
max_power_of_2 = 20

# We will dump our results to a checkpoint
chk = firedrake.DumbCheckpoint("poisson-inverse-conductivity-posterior-consistency-chk", mode=firedrake.FILE_CREATE)
np_file = open('poisson-inverse-conductivity-posterior-consistency-chk.npy', 'wb')

for i in range(min_power_of_2, max_power_of_2+1):

    print(f'i = {i}')

    # Setup Plot
    ukw = {'vmin': 0.0, 'vmax': +0.2}
    kw = {'vmin': -4, 'vmax': +4, 'shading': 'gouraud'}
    title_fontsize = 20
    text_fontsize = 20
    fig, axes = plt.subplots(ncols=3, nrows=1+len(methods), sharex=True, sharey=True, figsize=(20,30), dpi=200)
    plt.suptitle('Estimating Log-Conductivity $q$ \n    where $k = k_0e^q$ and $-\\nabla \\cdot k \\nabla u = f$ for known $f$', fontsize=title_fontsize)
    for ax in axes.ravel():
        ax.set_aspect('equal')

    # Plot True solution
    print('plotting true solution')
    # Column 0
    axes[0, 0].set_title('$u_{true}$', fontsize=title_fontsize)
    colors = firedrake.tripcolor(u_true, axes=axes[0, 0], shading='gouraud', **ukw)
    cax = make_axes_locatable(axes[0, 0]).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(colors, cax=cax)
    # Column 1
    axes[0, 1].set_title('$q_{true}$', fontsize=title_fontsize)
    colors = firedrake.tripcolor(q_true, axes=axes[0, 1], **kw)
    cax = make_axes_locatable(axes[0, 1]).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(colors, cax=cax)
    # Column 2
    axes[0, 2].set_title('$q_{true}-q_{true}$', fontsize=title_fontsize)
    zero_func = firedrake.Function(Q).assign(q_true-q_true)
    axes[0, 2].text(0.5, 0.5, f'$L^2$ Norm {firedrake.norm(zero_func, "L2"):.2f}', ha='center', fontsize=text_fontsize)
    colors = firedrake.tripcolor(zero_func, axes=axes[0, 2], **kw);
    cax = make_axes_locatable(axes[0, 2]).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(colors, cax=cax)

    # Make random point cloud
    num_points = 2**i
    xs = np.random.random_sample((num_points,2))

    # Generate "observed" data
    print(f'Generating {num_points} fake observed values')
    ζ = generator.standard_normal(len(xs))
    u_obs_vals = np.array(u_true.at(xs)) + float(σ) * ζ

    # save numpy arrays (just in case)
    print('Saving point cloud coordinates')
    np.save(np_file, xs)
    print('Saving u_obs_vals')
    np.save(np_file, u_obs_vals)

    for method_i, method in enumerate(methods):

        print(f'using {method} method')

        # Run the forward problem with q = 0 as first guess
        print('Running forward model')
        u = firedrake.Function(V)
        q = firedrake.Function(Q)
        bc = firedrake.DirichletBC(V, 0, 'on_boundary')
        F = (k0 * exp(q) * inner(grad(u), grad(v)) - f * v) * dx
        firedrake.solve(F == 0, u, bc)

        if method == 'point-cloud':

            # Store data on the point_cloud using a vertex only mesh
            print('Creating VertexOnlyMesh')
            point_cloud = firedrake.VertexOnlyMesh(mesh, xs)
            print('Creating P0DG(VertexOnlyMesh) space')
            P0DG = firedrake.FunctionSpace(point_cloud, 'DG', 0)
            print('Creating u_obs')
            u_obs = firedrake.Function(P0DG, name=f'u_obs_{method}_{num_points}')
            u_obs.dat.data[:] = u_obs_vals

            # Two terms in the functional
            misfit_expr = 0.5 * ((u_obs - firedrake.interpolate(u, P0DG)) / σ)**2
            α = firedrake.Constant(0.5)
            regularisation_expr = 0.5 * α**2 * inner(grad(q), grad(q))

        else:

            # Interpolating the mesh coordinates field (which is a vector function space)
            # into the vector function space equivalent of our solution space gets us
            # global DOF values (stored in the dat) which are the coordinates of the global
            # DOFs of our solution space. This is the necessary coordinates field X.
            print('Getting coordinates field X')
            Vc = firedrake.VectorFunctionSpace(mesh, V.ufl_element())
            X = firedrake.interpolate(mesh.coordinates, Vc).dat.data_ro[:]

            # Pick the appropriate "interpolate" method needed to create
            # u_interpolated given the chosen method
            print(f'Creating {method} interpolator')
            if method == 'nearest':
                interpolator = NearestNDInterpolator(xs, u_obs_vals)
            elif method == 'linear':
                interpolator = LinearNDInterpolator(xs, u_obs_vals, fill_value=0.0)
            elif method == 'clough-tocher':
                interpolator = CloughTocher2DInterpolator(xs, u_obs_vals, fill_value=0.0)
            elif method == 'gaussian':
                interpolator = Rbf(xs[:, 0], xs[:, 1], u_obs_vals, function='gaussian')
            print('Interpolating to create u_interpolated')
            u_interpolated = firedrake.Function(V, name=f'u_interpolated_{method}_{num_points}')
            u_interpolated.dat.data[:] = interpolator(X[:, 0], X[:, 1])

            # Two terms in the functional - note difference in misfit term!
            misfit_expr = 0.5 * ((u_interpolated - u) / σ)**2
            α = firedrake.Constant(0.5)
            regularisation_expr = 0.5 * α**2 * inner(grad(q), grad(q))

        # Should be able to write firedrake.assemble(misfit + regularisation * dx) but can't yet
        # because of the meshes being different in the point-cloud case
        print('Assembling J')
        J = firedrake.assemble(misfit_expr * dx) + firedrake.assemble(regularisation_expr * dx)

        # Create reduced functional
        print('Creating q̂ and Ĵ')
        q̂ = firedrake_adjoint.Control(q)
        Ĵ = firedrake_adjoint.ReducedFunctional(J, q̂)

        # Minimise reduced functional
        print('Minimising Ĵ to get q_min')
        q_min = firedrake_adjoint.minimize(
            Ĵ, method='Newton-CG', options={'disp': True}
        )
        q_min.rename(name=f'q_min_{method}_{num_points}')

        # Clear tape to avoid memory leak
        print('Clearing tape')
        tape.clear_tape()

        # Calculate error terms
        print('Calculating error')
        q_err = firedrake.Function(Q).assign(q_min-q_true)
        q_err.rename(name=f'q_err_{method}_{num_points}')
        print('Calculating error norm')
        l2norm = firedrake.norm(q_err, "L2")

        # Save results
        if method == 'point-cloud':
            # can remake vertex only mesh with point cloud coordinates!
            print('Checkpointing point cloud coordinates')
            chk.store(point_cloud.coordinates, name=f'point_cloud_coordinates_{num_points}')
            print('Checkpointing u_obs')
            chk.store(u_obs)
        else:
            print('Checkpointing u_interpolated')
            chk.store(u_interpolated)
        print('Checkpointing q_min')
        chk.store(q_min)
        print('Checkpointing q_err')
        chk.store(q_err)

        # Plot results

        row = method_i+1

        print(f'Plotting results for {method}')

        # column 0
        if method == 'point-cloud':
            axes[row, 0].set_title('Sampled Noisy $u_{obs}$', fontsize=title_fontsize)
            colors = axes[row, 0].scatter(xs[:, 0], xs[:, 1], c=u_obs_vals, vmin=0.0, vmax=0.2)
        else:
            axes[row, 0].set_title(f'$u_{{interpolated}}^{{{method}}}$', fontsize=title_fontsize)
            colors = firedrake.tripcolor(u_interpolated, axes=axes[row, 0], shading='gouraud', **ukw)
        cax = make_axes_locatable(axes[row, 0]).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(colors, cax=cax)

        # column 1
        if method == 'point-cloud':
            axes[row, 1].set_title('$q_{est}$ from $u_{obs}$', fontsize=title_fontsize)
        else:
            axes[row, 1].set_title(f'$q_{{est}}^{{{method}}}$ from $u_{{interpolated}}^{{{method}}}$', fontsize=title_fontsize)
        colors = firedrake.tripcolor(q_min, axes=axes[row, 1], **kw)
        cax = make_axes_locatable(axes[row, 1]).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(colors, cax=cax)

        # column 2
        axes[row, 2].set_title('$q_{est}-q_{true}$', fontsize=title_fontsize)
        axes[row, 2].text(0.5, 0.5, f'$L^2$ Norm {l2norm:.2f}', ha='center', fontsize=text_fontsize)
        colors = firedrake.tripcolor(q_err, axes=axes[row, 2], **kw);
        cax = make_axes_locatable(axes[row, 2]).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(colors, cax=cax)

    # Save figure
    print(f'Saving figure for {num_points} points')
    plt.savefig(f'posterior-consistency-{num_points}-pts.png')
