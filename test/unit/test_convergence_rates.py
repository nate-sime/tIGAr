from tIGAr.BSplines import *
import numpy as np


def test_poisson():
    n_eles = [8, 16, 32]
    L2_errors = np.zeros(len(n_eles))

    for level, n_ele in enumerate(n_eles):
        p, q = 3, 3
        Xi_u = uniform_knots(p, 0.0, 1.0, n_ele)
        Xi_v = uniform_knots(q, 0.0, 1.0, n_ele)

        spline_mesh = ExplicitBSplineControlMesh([p, q], [Xi_u, Xi_v])
        splineGenerator = EqualOrderSpline(1, spline_mesh)

        field = 0
        scalarSpline = splineGenerator.get_scalar_spline(field)
        for parametricDirection in [0, 1]:
            for side in [0, 1]:
                sideDofs = scalarSpline.get_side_dofs(parametricDirection, side)
                splineGenerator.add_zero_dofs(field, sideDofs)

        QUAD_DEG = 2 * max(p, q)
        spline = ExtractedSpline(splineGenerator, QUAD_DEG)

        u = TrialFunction(spline.V)
        v = TestFunction(spline.V)

        x = spline.spatial_coordinates()
        soln = sin(pi * x[0]) * sin(pi * x[1])
        f = -spline.div(spline.grad(soln))

        a = inner(spline.grad(u), spline.grad(v)) * spline.dx
        L = inner(f, v) * spline.dx
        u = Function(spline.V)
        spline.solve_linear_variational_problem(a == L, u)

        L2_errors[level] = assemble(((u - soln) ** 2) * spline.dx)**0.5

    rates = np.log(L2_errors[:-1]/L2_errors[1:])/np.log(2.0)

    assert np.all(np.abs(rates - 4.0) < 1e-1)