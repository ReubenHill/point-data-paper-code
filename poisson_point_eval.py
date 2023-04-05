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
    m = UnitSquareMesh(20, 20)
    V = FunctionSpace(m, family="CG", degree=2)
    v = TestFunction(V)
    u = Function(V)

    # Random forcing Function with values in [1, 2].
    f = RandomGenerator(PCG64(seed=0)).beta(V, 1.0, 2.0)

    bc = DirichletBC(V, 0, "on_boundary")
    F = (inner(grad(u), grad(v)) - f * v) * dx
    solve(F == 0, u, bc)

    point_cloud = VertexOnlyMesh(m, coords)
    P0DG = FunctionSpace(point_cloud, "DG", 0)

    return interpolate(u, P0DG)
