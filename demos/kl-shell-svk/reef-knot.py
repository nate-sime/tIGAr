"""
This example considers a simplified variant of the problem from 
Section 4.4.3 of the paper

  http://www.public.iastate.edu/~jmchsu/files/Kamensky_et_al-2017-CMAME.pdf

in which a nonlocal contact formulation---arguably a form of peridynamics---is 
used to penalize interpenetration of shell structures.  The required 
shell structure geometry, in the format of a Rhino T-spline (although it 
is in fact only two B-spline patches) is available (compressed) here:

  https://www.dropbox.com/s/irdhyral91mmock/knot.iga.tgz?dl=1

(Credit to Fei Xu for designing this geometry, using the Rhino 3D CAD 
software.)  While the contact formulation is largely orthogonal to IGA, and
can be reused mutatis mutandis for FE analysis, this demo illustrates 
the robustness of the K--L shell formulation using $C^1$ displacements.  

For simplicity, we have omitted the singular kernel, adaptive time stepping,
and specialized nonlinear solver of the cited reference, and just empirically 
selected a sufficiently-large penalty and sufficiently-small time step for the 
simulation to reach a steady state without structural self-intersection or
divergence of the basic Newton iteration.  However, due to the choice of this
naive nonlinear solver and "worst-case" uniform time step, the simulation 
takes several hours on a modern workstation to reach an interesting 
configuration.  

The implementation here relies on a number of seemingly-fragile assumptions
regarding the ordering of DoFs in mixed function spaces, as pointed out in
the comments; we invite any suggestions for more robust indexing schemes.  
"""

from tIGAr import *

if(mpisize > 1):
    print("ERROR: This demo only works in serial.")
    exit()

# Check for existence of the required data file.
FNAME = "knot.iga"
import os.path
if(not os.path.isfile(FNAME)):
    if(mpirank==0):
        print("ERROR: The required input file '"+FNAME
              +"' is not present in the working directory. "
              +"Please refer to the docstring at the top of this script.")
    exit()

from tIGAr.RhinoTSplines import *
from numpy import zeros
from scipy.spatial import cKDTree
from numpy.linalg import norm as npNorm
from numpy import outer as npOuter
from numpy import identity as npIdentity

ADD_MODE = PETSc.InsertMode.ADD


####### Preprocessing #######

if(mpirank==0):
    print("Generating extraction data...")

# Load a control mesh from element-by-element extraction data in the file
# "knot.iga", which is generated by the T-spline plugin for Rhino 3D.
controlMesh = RhinoTSplineControlMesh(FNAME)

# Assume each component of the shell structure's displacement is discretized
# using the same scalar discrete space used for the components of the
# mapping from parametric to physical space.
d = 3
splineGenerator = EqualOrderSpline(d,controlMesh)

# Fix the left ends of the ribbons.  (Forces will be applied to the right
# ends, in the variational formulation below.)
class BdryDomain(SubDomain):
    def inside(self,x,on_boundary):
        return x[0] < -6.0
for i in range(0,d):
    splineGenerator.add_zero_dofs_by_location(BdryDomain(), i)

# Fix only the y- and z- components of displacement for the right ends.
class BdryDomain(SubDomain):
    def inside(self,x,on_boundary):
        return x[0] > 10.0
for i in range(1,d):
    splineGenerator.add_zero_dofs_by_location(BdryDomain(), i)
    
# Write the extraction data.
DIR = "./extraction"
splineGenerator.write_extraction(DIR)


####### Analysis #######

if(mpirank==0):
    print("Forming extracted spline...")

# Read an extracted spline back in.
QUAD_DEG = 6
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

if(mpirank==0):
    print("Starting analysis...")

# Potentially-fragile assumption on the ordering of DoFs in mixed-element
# function spaces:
def nodeToDof(node,direction):
    return d*node + direction

# Potentially-fragile assumption: that there is a correspondence in DoF order
# between the scalar space used for each component of the control mapping and
# the mixed space used for the displacement.  
nNodes = spline.cp_funcs[0].vector().get_local().size
nodeX = zeros((nNodes,d))
for i in range(0,nNodes):
    wi = spline.cp_funcs[d].vector().get_local()[i]
    for j in range(0,d):
        Xj_hom = spline.cp_funcs[j].vector().get_local()[i]
        nodeX[i,j] = Xj_hom/wi

# The contact potential is NOT defined in UFL; contact forces will be computed
# numerically and added to PETSc objects, which will then be added to the
# FEniCS-assembled FE matrices and vectors, prior to extraction.

# Unlike the original reference, we simply use a linear penalty force here,
# to avoid complicated adaptive time stepping and specialized nonlinear
# solvers.
r_max = 0.15
k = 1e8
def phiPrime(r):
    if(r>r_max):
        return 0.0
    return -k*(r_max-r)
def phiDoublePrime(r):
    if(r>r_max):
        return 0.0
    return k

# Using quadrature points coincident with the FE nodes of the extracted
# representation of the spline significantly simplifies the assembly process.
W = assemble(inner(Constant(d*(1.0,)),TestFunction(spline.V))*spline.dx)
quadWeightsTemp = W.get_local()
quadWeights = zeros(nNodes)
for i in range(0,nNodes):
    quadWeights[i] = quadWeightsTemp[nodeToDof(i,0)]

# Points closer together than this distance in the reference configuration
# do not interact through contact forces.
R_self = 0.4

# Overkill preallocation for the contact tangent matrix; if a node interacts
# with too many other nodes, this could be exceeded, and the contact force
# assembly will slow down drastically.
PREALLOC = 500

def assembleContact(dispFunc):
    """
    Return FE stiffness matrix and load vector contributions associated with
    contact forces, based on an FE displacement ``dispFunc``.  Note that this
    contact assembly is largely orthogonal to IGA, and, aside from the
    incorporation of weights into the displacements, could be re-used in the
    pure FE setting.
    """

    # Establish tensors to accumulate contact contributions.
    F = assemble(inner(Constant(d*(0.0,)),TestFunction(spline.V))*dx,
                 finalize_tensor=False)
    Fv = as_backend_type(F).vec()
    KPETSc = PETSc.Mat()
    KPETSc.createAIJ([[d*nNodes,None],[None,d*nNodes]])
    KPETSc.setPreallocationNNZ([PREALLOC,PREALLOC])
    KPETSc.setUp()
    K = PETScMatrix(KPETSc)
    Km = as_backend_type(K).mat()

    # Ideally, we would first examine the set of pairs
    # returned by the cKDTree query, then allocate based
    # on that, rather than the present approach of
    # preallocating a large number of nonzeros and hoping
    # for the best.  
    Km.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,False)

    # Obtain displacement in homogeneous representation.
    dispFlat = dispFunc.vector().get_local()
    disp = dispFlat.reshape((-1,d))
    # Divide nodal displacements through by FE nodal weights.
    for i in range(0,nNodes):
        wi = spline.cp_funcs[d].vector().get_local()[i]
        for j in range(0,d):
            disp[i,j] /= wi
    # Compute deformed positions of nodes in physical space.
    nodex = nodeX+disp

    tree = cKDTree(nodex)
    pairs = tree.query_pairs(r_max,output_type='ndarray')

    # Because the ndarray output from the scipy cKDTree maps onto a C++ type,
    # this loop could likely be optimized by placing it in a JIT-compiled
    # C++ extension module.  
    for pair in pairs:

        node1 = pair[0]
        node2 = pair[1]

        # Positions of the two nodal quadrature points in the reference
        # configuration:
        X1 = nodeX[node1,:]
        X2 = nodeX[node2,:]
        R12 = npNorm(X2-X1)

        # Do not add contact forces between points that are too close in the
        # reference configuration.  (Otherwise, the entire structure would
        # expand, trying to get away from itself.)
        if(R12 > R_self):

            # Positions of nodes in the current configuration:
            x1 = nodex[node1,:]
            x2 = nodex[node2,:]

            # Force computation: see (24) from original reference.
            r12vec = x2-x1
            r12 = npNorm(r12vec)
            r12hat = r12vec/r12
            r_otimes_r = npOuter(r12hat,r12hat)
            I = npIdentity(d)
            C = quadWeights[node1]*quadWeights[node2]
            f12 = C*phiPrime(r12)*r12hat

            # Nodal FE spline (not quadrature) weights:
            w1 = spline.cp_funcs[d].vector().get_local()[node1]
            w2 = spline.cp_funcs[d].vector().get_local()[node2]

            # Add equal-and-opposite forces to the RHS vector.
            for direction in range(0,d):
                dof1 = nodeToDof(node1,direction)
                dof2 = nodeToDof(node2,direction)
                Fv.setValue(dof1,-f12[direction]/w1,addv=ADD_MODE)
                Fv.setValue(dof2,f12[direction]/w2,addv=ADD_MODE)
                # (Weights are involved here because the FE test function
                # that goes to 1 at a node is in homogeneous representation.)

            # Tangent computation: see (25)--(26) from original reference.  
            k12_tensor = C*(phiDoublePrime(r12)*r_otimes_r \
                            + (phiPrime(r12)/r12)*(I-r_otimes_r))

            # Add tangent contributions to the LHS matrix.
            for d1 in range(0,d):
                for d2 in range(0,d):
                    n1dof1 = nodeToDof(node1,d1)
                    n1dof2 = nodeToDof(node1,d2)
                    n2dof1 = nodeToDof(node2,d1)
                    n2dof2 = nodeToDof(node2,d2)
                    k12 = k12_tensor[d1,d2]
            
                    # 11 contribution:
                    Km.setValue(n1dof1,n1dof2,k12/(w1*w1),addv=ADD_MODE)
                    # 22 contribution:
                    Km.setValue(n2dof1,n2dof2,k12/(w2*w2),addv=ADD_MODE)
                    # Off-diagonal contributions:
                    Km.setValue(n1dof1,n2dof2,-k12/(w1*w2),addv=ADD_MODE)
                    Km.setValue(n2dof1,n1dof2,-k12/(w1*w2),addv=ADD_MODE)
                    # (Weights are involved here because FE test and trial
                    # space basis functions that go to 1 at nodes are in
                    # homogeneous representation.)
    Fv.assemble()
    Km.assemble()

    return K,F

# Displacement solution at current and previous time steps:
DELTA_T = Constant(0.001)
y_hom = Function(spline.V)
y_old_hom = Function(spline.V)
ydot_hom = Constant(1.0/DELTA_T)*y_hom+Constant(-1.0/DELTA_T)*y_old_hom
ydot_old_hom = Function(spline.V)
yddot_hom = (ydot_hom-ydot_old_hom)/DELTA_T

# Displacement solution and time derivatives in rational form:
y = spline.rationalize(y_hom)
ydot = spline.rationalize(ydot_hom)
yddot = spline.rationalize(yddot_hom)

# Reference and deformed configurations:
X = spline.F
x = X + y

# Helper function to normalize a vector v.
def unit(v):
    return v/sqrt(inner(v,v))

# Helper function to compute geometric quantities for a midsurface
# configuration x.
def shellGeometry(x):

    # Covariant basis vectors:
    dxdxi = spline.parametric_grad(x)
    a0 = as_vector([dxdxi[0,0],dxdxi[1,0],dxdxi[2,0]])
    a1 = as_vector([dxdxi[0,1],dxdxi[1,1],dxdxi[2,1]])
    a2 = unit(cross(a0,a1))

    # Metric tensor:
    a = as_matrix(((inner(a0,a0),inner(a0,a1)),
                   (inner(a1,a0),inner(a1,a1))))
    # Curvature:
    deriva2 = spline.parametric_grad(a2)
    b = -as_matrix(((inner(a0,deriva2[:,0]),inner(a0,deriva2[:,1])),
                    (inner(a1,deriva2[:,0]),inner(a1,deriva2[:,1]))))
    
    return (a0,a1,a2,a,b)

# Use the helper function to obtain shell geometry for the reference
# and current configurations defined earlier.
A0,A1,A2,A,B = shellGeometry(X)
a0,a1,a2,a,b = shellGeometry(x)

# Strain quantities.
epsilon = 0.5*(a - A)
kappa = B - b

# Helper function to convert a 2x2 tensor T to its local Cartesian
# representation, in a shell configuration with metric a, and covariant
# basis vectors a0 and a1.
def cartesian(T,a,a0,a1):
    
    # Raise the indices on the curvilinear basis to obtain contravariant
    # basis vectors a0c and a1c.
    ac = inv(a)
    a0c = ac[0,0]*a0 + ac[0,1]*a1
    a1c = ac[1,0]*a0 + ac[1,1]*a1

    # Perform Gram--Schmidt orthonormalization to obtain the local Cartesian
    # basis vector e0 and e1.
    e0 = unit(a0)
    e1 = unit(a1 - e0*inner(a1,e0))

    # Perform the change of basis on T and return the result.
    ea = as_matrix(((inner(e0,a0c),inner(e0,a1c)),
                    (inner(e1,a0c),inner(e1,a1c))))
    ae = ea.T
    return ea*T*ae

# Use the helper function to compute the strain quantities in local
# Cartesian coordinates.
epsilonBar = cartesian(epsilon,A,A0,A1)
kappaBar = cartesian(kappa,A,A0,A1)

# Helper function to convert a 2x2 tensor to voigt notation, following the
# convention for strains, where there is a factor of 2 applied to the last
# component.  
def voigt(T):
    return as_vector([T[0,0],T[1,1],2.0*T[0,1]])

# The Young's modulus and Poisson ratio:
E = Constant(1e7)
nu = Constant(0.3)

# The material matrix:
D = (E/(1.0 - nu*nu))*as_matrix([[1.0,  nu,   0.0         ],
                                 [nu,   1.0,  0.0         ],
                                 [0.0,  0.0,  0.5*(1.0-nu)]])
# The shell thickness:
h_th = Constant(0.004)

# Extension and bending resultants:
nBar = h_th*D*voigt(epsilonBar)
mBar = (h_th**3)*D*voigt(kappaBar)/12.0

# Compute the elastic potential energy density
Wint = 0.5*(inner(voigt(epsilonBar),nBar)
            + inner(voigt(kappaBar),mBar))*spline.dx

# Take the Gateaux derivative of Wint(y) in the direction of the test
# function z to obtain the internal virtual work.  Because y is not a
# Function, and therefore not a valid argument to derivative(), we take
# the derivative w.r.t. y_hom, in the direction z_hom, which is equivalent.
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)
dWint = derivative(Wint,y_hom,z_hom)

# Mass density:
DENS = Constant(1.0)

# Inertial contribution to the residual, including a body force on the right
# ends of the ribbons.
bodyForceMag = conditional(gt(X[0],10.0),Constant(1e3),Constant(0.0))
bodyForce = as_vector([bodyForceMag,Constant(0.0),Constant(0.0)])
dWmass = DENS*h_th*inner(yddot-bodyForce,z)*spline.dx

# Mass damping:
DAMP = Constant(1.0e0)
dWdamp = DAMP*DENS*h_th*inner(ydot,z)*spline.dx

# The full nonlinear residual for the shell problem:
res = dWmass + dWint + dWdamp

# Use derivative() to obtain the consistent tangent of the nonlinear residual,
# considered as a function of displacement in homogeneous coordinates.
Dres = derivative(res,y_hom)

# Settings for the time stepping and Newton iteration:
N_TIME_STEPS = 3000
MAX_ITERS = 100
REL_TOL = 1e-2

# For x, y, and z components of displacement:
d0File = File("results/disp-x.pvd")
d1File = File("results/disp-y.pvd")
d2File = File("results/disp-z.pvd")

# For x, y, and z components of initial configuration:
F0File = File("results/F-x.pvd")
F1File = File("results/F-y.pvd")
F2File = File("results/F-z.pvd")

# For weights:
F3File = File("results/F-w.pvd")

# Number of time steps per output file batch:
OUTPUT_SKIP = 5

# Time stepping loop:
for timeStep in range(0,N_TIME_STEPS):

    print("------- Time step "+str(timeStep)+" -------")

    # Output fields needed for visualization.
    if(timeStep % OUTPUT_SKIP == 0):
        (d0,d1,d2) = y_hom.split()
        d0.rename("d0","d0")
        d1.rename("d1","d1")
        d2.rename("d2","d2")
        d0File << d0
        d1File << d1
        d2File << d2
        # (Note that the components of spline.F are rational, and cannot be
        # directly outputted to ParaView files.)
        spline.cp_funcs[0].rename("F0", "F0")
        spline.cp_funcs[1].rename("F1", "F1")
        spline.cp_funcs[2].rename("F2", "F2")
        spline.cp_funcs[3].rename("F3", "F3")
        F0File << spline.cp_funcs[0]
        F1File << spline.cp_funcs[1]
        F2File << spline.cp_funcs[2]
        F3File << spline.cp_funcs[3]
    

    # Because of the non-standard assembly process, in which contributions
    # not coming from a UFL Form are directly added to the residual and
    # tangent matrix, the Newton iteration has been implemented manually
    # in this example.  
    for newtonStep in range(0,MAX_ITERS):

        # First, assemble the contributions coming from UFL Forms.
        K = assemble(Dres)
        F = assemble(res)

        # Next, add on the contact contributions, assembled using the
        # function defined above.
        Kc,Fc = assembleContact(y_hom)
        K += Kc
        F += Fc

        # Apply the extraction to an IGA function space.  (This applies
        # the Dirichlet BCs on the IGA unknowns.)
        MTKM = spline.extract_matrix(K)
        MTF = spline.extract_vector(F)

        # Check the nonlinear residual.
        Fnorm = norm(MTF)
        if(newtonStep==0):
            Fnorm0 = Fnorm
        relNorm = Fnorm/Fnorm0
        
        print("  ....... Newton step "+str(newtonStep)
              +" : relative residual = "+str(relNorm))

        # Solve for the nonlinear increment, and add it to the current
        # solution guess.
        dy_hom = Function(spline.V)
        spline.solveLinearSystem(MTKM,MTF,dy_hom)
        y_hom.assign(y_hom-dy_hom)

        if(relNorm < REL_TOL):
            break
        if(newtonStep == MAX_ITERS-1):
            print("ERROR: Nonlinear solution diverged.")
            exit()
        
    # Move to the next time step.
    ydot_old_hom.assign(ydot_hom)
    y_old_hom.assign(y_hom)


####### Postprocessing #######

# Notes for plotting the results with ParaView:
#
# Load the time series from all seven files and combine them with the
# Append Attributes filter.  Then use the Calculator filter to define the
# vector field
#
# ((d0+F0)/F3-coordsX)*iHat+((d1+F1)/F3-coordsY)*jHat+((d2+F2)/F3-coordsZ)*kHat
#
# which can then be used in the Warp by Vector filter.  Because the
# parametric domain is artificially stretched out, the result of the Warp by
# Vector filter will be much smaller, and the window will need to be re-sized
# to fit the warped data.  The scale factor on the warp filter may need to
# manually be set to 1.
