from firedrake import *


def poisson_point_eval(coords):
    """Solve Poisson's equation on a unit square for a random forcing term
    with Firedrake and evaluate at a user-specified set of point coordinates.

    Parameters
    ----------
    coords: numpy.ndarray
        A point coordinates array of shape (N, 2) to evaluate the solution at.

    Returns
    -------
    firedrake.function.Function
        A field containing the point evaluatations.
    """
    omega = UnitSquareMesh(20, 20)
    P2CG = FunctionSpace(omega, family="CG", degree=2)
    u = Function(P2CG)
    v = TestFunction(P2CG)

    # Random forcing Function with values in [1, 2].
    f = RandomGenerator(PCG64(seed=0)).beta(P2CG, 1.0, 2.0)

    F = (inner(grad(u), grad(v)) - f * v) * dx
    bc = DirichletBC(P2CG, 0, "on_boundary")
    solve(F == 0, u, bc)

    omega_v = VertexOnlyMesh(omega, coords)
    P0DG = FunctionSpace(omega_v, "DG", 0)

    return interpolate(u, P0DG)
