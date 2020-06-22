"""
The "hello, world" of computational PDEs:  Solve the Poisson equation, 
verifying accuracy via the method of manufactured solutions.  

This example uses the simplest IGA discretization, namely, explicit B-splines
in which parametric and physical space are the same.
"""

import numpy as np
from tIGAr import *
from tIGAr.BSplines import *
import math

n_eles = [4, 8, 16, 32]

# Array to store error at different refinement levels:
L2_errors = np.zeros(len(n_eles))

for level, n_ele in enumerate(n_eles):
    p, q = 3, 3
    Xi_u = uniform_knots(p, 0.0, 1.0, n_ele)
    Xi_v = uniform_knots(q, 0.0, 1.0, n_ele)

    # Create a control mesh for which D = D̂.
    spline_mesh = ExplicitBSplineControlMesh([p, q], [Xi_u, Xi_v])

    # Create a spline generator for a spline with a single scalar field on the
    # given control mesh, where the scalar field is the same as the one used
    # to determine the mapping $\mathbf{F}:\widehat{\Omega}\to\Omega$.
    spline_generator = EqualOrderSpline(1, spline_mesh)

    # Set Dirichlet boundary conditions on the 0-th (and only) field, on both
    # ends of the domain, in both directions.
    field = 0
    scalar_spline = spline_generator.get_scalar_spline(field)
    for parametricDirection in [0, 1]:
        for side in [0, 1]:
            sideDofs = scalar_spline.get_side_dofs(parametricDirection, side)
            spline_generator.add_zero_dofs(field, sideDofs)

    # Alternative: set BCs based on location of corresponding control points.
    # (Note that this only makes sense for splineGenerator of type
    # EqualOrderSpline; for non-equal-order splines, there is not
    # a one-to-one correspondence between degrees of freedom and geometry
    # control points.)

    # field = 0
    # class BdryDomain(SubDomain):
    #    def inside(self,x,on_boundary):
    #        return (near(x[0],x0) or near(x[0],x0+Lx)
    #                or near(x[1],y0) or near(x[1],y0+Ly))
    # splineGenerator.addZeroDofsByLocation(BdryDomain(),field)

    # Write extraction data to the filesystem.
    DIR = "./extraction"
    spline_generator.writeExtraction(DIR)

    ####### Analysis #######

    if (mpirank == 0):
        print("Setting up extracted spline...")

    # Choose the quadrature degree to be used throughout the analysis.
    # In IGA, especially with rational spline spaces, under-integration is a
    # fact of life, but this does not impair optimal convergence.
    QUAD_DEG = 2 * max(p, q)

    # Create the extracted spline directly from the generator.
    # As of version 2019.1, this is required for using quad/hex elements in
    # parallel.
    spline = ExtractedSpline(spline_generator, QUAD_DEG)

    # Alternative: Can read the extracted spline back in from the filesystem.
    # For quad/hex elements, in version 2019.1, this only works in serial.

    # spline = ExtractedSpline(DIR,QUAD_DEG)

    if (mpirank == 0):
        print("Solving...")

    # Homogeneous coordinate representation of the trial function u.  Because 
    # weights are 1 in the B-spline case, this can be used directly in the PDE,
    # without dividing through by weight.
    u = TrialFunction(spline.V)

    # Corresponding test function.
    v = TestFunction(spline.V)

    # Create a force, f, to manufacture the solution, soln
    x = spline.spatialCoordinates()
    soln = sin(pi * x[0]) * sin(pi * x[1])
    f = -spline.div(spline.grad(soln))

    # Set up and solve the Poisson problem
    a = inner(spline.grad(u), spline.grad(v)) * spline.dx
    L = inner(f, v) * spline.dx
    u = Function(spline.V)
    spline.solveLinearVariationalProblem(a == L, u)

    ####### Postprocessing #######

    # The solution, u, is in the homogeneous representation, but, again, for
    # B-splines with weight=1, this is the same as the physical representation.
    File("results/u.pvd") << u

    # Compute and print the $L^2$ error in the discrete solution.
    L2_error = math.sqrt(assemble(((u - soln) ** 2) * spline.dx))
    L2_errors[level] = L2_error
    if (level > 0):
        rate = math.log(L2_errors[level - 1] / L2_errors[level]) / math.log(2.0)
    else:
        rate = "--"
    if (mpirank == 0):
        print("L2 Error for level " + str(level) + " = " + str(L2_error)
              + "  (rate = " + str(rate) + ")")
