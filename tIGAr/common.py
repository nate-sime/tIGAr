"""
The ``common`` module 
---------------------
contains basic definitions of abstractions for 
generating extraction data and importing it again for use in analysis.  Upon
importing this module, a number of setup steps are carried out 
(e.g., initializing MPI).
"""

import functools
import typing
import numpy
import abc
import scipy.stats

import dolfin

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

import ufl.equation


worldcomm = dolfin.MPI.comm_world
selfcomm = dolfin.MPI.comm_self

mpisize = dolfin.MPI.size(worldcomm)
mpirank = dolfin.MPI.rank(worldcomm)

from tIGAr.calculusUtils import *

INDEX_TYPE = 'int32'
#DEFAULT_PREALLOC = 100
DEFAULT_PREALLOC = 500

# Choose default behavior for permutation of indices based on the number
# of MPI tasks
if(mpisize > 8):
    DEFAULT_DO_PERMUTATION = True
else:
    DEFAULT_DO_PERMUTATION = False

# basis function evaluations less than this will be considered outside the
# function's support
DEFAULT_BASIS_FUNC_IGNORE_EPS = 1e-15

# This was too small for optimal convergence rates in high-order biharmonic
# discretizations with highly-refined meshes:
#DEFAULT_BASIS_FUNC_IGNORE_EPS = 1e-9


# file naming conventions
EXTRACTION_DATA_FILE = "extraction-data.h5"
EXTRACTION_INFO_FILE = "extraction-info.txt"
EXTRACTION_H5_MESH_NAME = "/mesh"
def EXTRACTION_H5_CONTROL_FUNC_NAME(dim):
    return "/control"+str(dim)
#def EXTRACTION_ZERO_DOFS_FILE(proc):
#    return "/zero-dofs"+str(proc)+".dat"
EXTRACTION_ZERO_DOFS_FILE = "zero-dofs.dat"
EXTRACTION_MAT_FILE = "extraction-mat.dat"
EXTRACTION_MAT_FILE_CTRL = "extraction-mat-ctrl.dat"

# DG space is more memory-hungry, but allows for $C^{-1}$-continuous splines,
# e.g., for div-conforming VMS, and will still work for more continuous
# spaces.  Right now, it is not supported for quad/hex elements.
USE_DG_DEFAULT = True

# whether or not to use tensor product elements by default
USE_RECT_ELEM_DEFAULT = True

# whether or not to explicitly form M^T (memory vs. speed tradeoff)
FORM_MT = False

# Helper function to generate unique temporary file names for DOLFIN
# XML meshes; file name is unique for a given rank on a given communicator.
def generateMeshXMLFileName(comm):
    import hashlib
    s = repr(comm)+repr(dolfin.MPI.rank(comm))
    return "mesh-"\
        +str(hashlib.md5(s.encode("utf-8"))\
             .hexdigest())+".xml"

# helper function to do MatMultTranspose() without all the setup steps for the
# results vector
def multTranspose(M,b):
    """
    Returns ``M^T*b``, where ``M`` and ``b`` are DOLFIN ``GenericTensor`` and
    ``GenericVector`` objects.
    """
    totalDofs = dolfin.as_backend_type(M).mat().getSizes()[1][1]
    comm = dolfin.as_backend_type(M).mat().getComm()
    MTbv = PETSc.Vec(comm)
    MTbv.create(comm=comm)
    MTbv.setSizes(totalDofs)
    MTbv.setUp()
    dolfin.as_backend_type(M).mat().multTranspose(
        dolfin.as_backend_type(b).vec(), MTbv)
    return dolfin.PETScVector(MTbv)


# helper function to generate an identity permutation IS 
# given an ownership range
def generateIdentityPermutation(ownRange,comm=worldcomm):

    """
    Returns a PETSc index set corresponding to the ownership range.
    """

    iStart = ownRange[0]
    iEnd = ownRange[1]
    localSize = iEnd - iStart
    iArray = numpy.zeros(localSize,dtype=INDEX_TYPE)
    for i in numpy.arange(0,localSize):
        iArray[i] = i+iStart
    retval = PETSc.IS(comm)
    retval.createGeneral(iArray,comm=comm)
    return retval

class AbstractExtractionGenerator(abc.ABC):

    """
    Abstract class representing the minimal set of functions needed to write
    extraction operators for a spline.
    """

    def __init__(self, comm):

        """
        Arguments in ``*args`` are passed as a tuple to 
        ``self.customSetup()``.  Appropriate arguments vary by subclass.  If
        the first argument ``comm`` is of type ``petsc4py.PETSc.Comm``, then
        it will be treated as a communicator for the extraction generator.
        Otherwise, it is treated as if it were the first argument in ``args``.
        """
        self.comm = comm
        self.mesh = self.generate_mesh()

        # note: if self.nsd is set in a customSetup, then the subclass
        # getNsd() references that, this is still safe
        self.nsd = self.get_nsd()

        self.VE_control = ufl.FiniteElement(self.extraction_element(),
                                        self.mesh.ufl_cell(),
                                        self.get_degree(-1))
        self.V_control = dolfin.FunctionSpace(self.mesh, self.VE_control)

        if self.get_num_fields() > 1:
            VE_components = []
            for i in range(0, self.get_num_fields()):
                VE_components \
                    += [ufl.FiniteElement(self.extraction_element(),
                                      self.mesh.ufl_cell(),
                                      self.get_degree(i)), ]

            self.VE = ufl.MixedElement(tuple(VE_components))
        else:
            self.VE = ufl.FiniteElement(self.extraction_element(),
                                    self.mesh.ufl_cell(),
                                    self.get_degree(0))

        self.V = dolfin.FunctionSpace(self.mesh, self.VE)

        self.cpFuncs = []
        for i in range(0, self.nsd + 1):
            self.cpFuncs += [dolfin.Function(self.V_control), ]

        self.M_control = self.generateM_control()
        self.M = self.generateM()

        # get transpose
        if (FORM_MT):
            MT_control = dolfin.PETScMatrix(self.M_control.mat()
                                     .transpose(PETSc.Mat(self.comm)))

        # generating CPs, weights in spline space:
        # (control net never permuted)
        for i in range(0, self.nsd + 1):
            if (FORM_MT):
                MTC = MT_control * (self.cpFuncs[i].vector())
            else:
                MTC = multTranspose(self.M_control, self.cpFuncs[i].vector())
            Istart, Iend = dolfin.as_backend_type(MTC).vec().getOwnershipRange()
            for I in numpy.arange(Istart, Iend):
                dolfin.as_backend_type(MTC).vec()[I] \
                    = self.get_homogeneous_coordinate(I, i)
            dolfin.as_backend_type(MTC).vec().assemblyBegin()
            dolfin.as_backend_type(MTC).vec().assemblyEnd()

            self.cpFuncs[i].vector().set_local(
                (self.M_control * MTC).get_local())
            dolfin.as_backend_type(self.cpFuncs[i].vector()).vec().ghostUpdate()

        # may need to be permuted
        self.zero_dofs = []  # self.generateZeroDofs()

        # replace M with permuted version
        # if(mpisize > 1):
        #
        #    self.permutation = self.generatePermutation()
        #    newM = self.M.mat()\
        #                 .permute\
        #                 (generateIdentityPermutation\
        #                  (self.M.mat().getOwnershipRange()),\
        #                  self.permutation)
        #    self.M = PETScMatrix(newM)
        #
        #    # fix list of zero DOFs
        #    self.permutationAO = PETSc.AO()
        #    self.permutationAO\
        #        .createBasic(self.permutation,\
        #                     generateIdentityPermutation\
        #                     (self.M.mat().getOwnershipRangeColumn()))
        #    zeroDofIS = PETSc.IS()
        #    zeroDofIS.createGeneral(array(self.zeroDofs,dtype=INDEX_TYPE))
        #    self.zeroDofs = self.permutationAO.app2petsc\
        #                    (zeroDofIS).getIndices()

    def getComm(self):
        """
        Returns the extraction generator's MPI communicator.
        """
        return self.comm

    # what type of element (CG or DG) to extract to
    # (override in subclass for non-default behavior)
    def use_dg(self) -> bool:
        """
        Returns
        -------
        Indicate whether or not to use DG elements in extraction.
        """
        return USE_DG_DEFAULT

    def extraction_element(self):
        """
        Returns
        -------
        Indicate what type of finite element to use in extraction.
        """
        return "DG" if self.use_dg() else "Lagrange"

    @abc.abstractmethod
    def get_num_fields(self) -> int:
        """
        Returns
        -------
        The number of unknown fields for the spline.
        """
        pass

    @abc.abstractmethod
    def get_homogeneous_coordinate(self, node: int, direction: int):
        """
        Get the homogeneous coordinate of a specified spline control point node
        in the given direction.

        Parameters
        ----------
        node: Spline control point node index
        direction: Homogeneous coordinate index
        """
        pass

    @abc.abstractmethod
    def generate_mesh(self) -> dolfin.Mesh:
        """
        Generate and return an FE mesh suitable for extracting the spline space.

        Returns
        -------
        Corresponding DOLFIN FE mesh
        """
        pass

    @abc.abstractmethod
    def get_degree(self, field: int) -> int:
        """
        Parameters
        ----------
        field: Field index. ``-1`` corresponds to the degree of the control
            field.

        Returns
        -------
        Degree of polynomial in extracted representation.
        """
        pass

    @abc.abstractmethod
    def get_ncp(self, field: int) -> int:
        """
        Parameters
        ----------
        field: Field index. ``-1`` corresponds to the control mesh field.

        Returns
        -------
        Total number of degrees of freedom of the given field.
        """
        pass

    @abc.abstractmethod
    def get_nsd(self):
        """
        Return the number of spatial dimensions of the physical domain.
        """
        pass

    def globalDof(self, field, localDof):
        """
        Given a ``field`` and a local DoF number ``localDof``, 
        return the global DoF number; 
        this is BEFORE any re-ordering for parallelization.
        """
        # offset localDof by 
        retval = localDof
        for i in range(0, field):
            retval += self.get_ncp(i)
        return retval

    def generatePermutation(self):
        """
        Generates an index set to permute the IGA degrees of freedom
        into an order that is (hopefully) efficient given the partitioning
        of the FEM nodes.  Assume that ``self.M`` currently holds the
        un-permuted extraction matrix.
        Default implementation just fills in an identity permutation.
        """
        return generateIdentityPermutation\
            (self.M.mat().getOwnershipRangeColumn(),self.comm)

    def add_zero_dofs_global(self, new_dofs: typing.List[int]):
        """
        Add new DoFs in global numerical to the list of DoFs to which
        homogeneous BCs will be applied.

        Parameters
        ----------
        new_dofs: Global numbering of new DoFs
        """
        self.zero_dofs += new_dofs

    def add_zero_dofs(self, field: int, new_dofs: typing.List[int]):
        """
        Apply homogeneous Dirichlet BCs to the provided DoFs.

        Parameters
        ----------
        field: Field index
        new_dofs: Local numbering of new DoFs
        """
        new_dofs_global = new_dofs[:]
        for i in range(0, len(new_dofs)):
            new_dofs_global[i] = self.globalDof(field, new_dofs[i])
        self.add_zero_dofs_global(new_dofs_global)

    def get_prealloc(self, control: bool) -> int:
        """
        Parameters
        ----------
        control: Indicating whether or not this is the preallocation for the
            scalar field used for control point coordinates.

        Returns
        -------
        The number of entries per row needed in the extraction matrix

        Note
        ----
        If left as the default, this could potentially slow down drastically
        for very high-order splines, or waste a lot of memory for low order
        splines. In general, it is a good idea to override this in subclasses.
        """
        return DEFAULT_PREALLOC

    def getIgnoreEps(self):
        """
        Returns an absolute value below which basis function evaluations are
        considered to be outside of the function's support.  

        This method is very unlikely to require overriding in subclasses.
        """
        return DEFAULT_BASIS_FUNC_IGNORE_EPS

    @abc.abstractmethod
    def generateM_control(self):
        """
        Return the extraction matrix for the control field.
        """
        pass

    @abc.abstractmethod
    def generateM(self):
        """
        Return the extraction matrix for the unknowns.
        """
        pass

    # def genericSetup(self):
    #     """
    #     Common setup steps for all subclasses (called in ``self.__init__()``).
    #     """
        #
        # self.mesh = self.generateMesh()
        #
        # # note: if self.nsd is set in a customSetup, then the subclass
        # # getNsd() references that, this is still safe
        # self.nsd = self.getNsd()
        #
        # self.VE_control = FiniteElement(self.extractionElement(),\
        #                                 self.mesh.ufl_cell(),\
        #                                 self.get_degree(-1))
        # self.V_control = FunctionSpace(self.mesh,self.VE_control)
        #
        # if(self.getNFields() > 1):
        #     VE_components = []
        #     for i in range(0,self.getNFields()):
        #         VE_components \
        #             += [FiniteElement(self.extractionElement(),\
        #                               self.mesh.ufl_cell(),\
        #                               self.get_degree(i)),]
        #
        #     self.VE = MixedElement(tuple(VE_components))
        # else:
        #     self.VE = FiniteElement(self.extractionElement(),\
        #                             self.mesh.ufl_cell(),\
        #                             self.get_degree(0))
        #
        # self.V = FunctionSpace(self.mesh,self.VE)
        #
        # self.cpFuncs = []
        # for i in range(0,self.nsd+1):
        #     self.cpFuncs += [Function(self.V_control),]
        #
        # self.M_control = self.generateM_control()
        # self.M = self.generateM()
        #
        # # get transpose
        # if(FORM_MT):
        #     MT_control = PETScMatrix(self.M_control.mat()
        #                              .transpose(PETSc.Mat(self.comm)))
        #
        # # generating CPs, weights in spline space:
        # # (control net never permuted)
        # for i in range(0,self.nsd+1):
        #     if(FORM_MT):
        #         MTC = MT_control*(self.cpFuncs[i].vector())
        #     else:
        #         MTC = multTranspose(self.M_control,self.cpFuncs[i].vector())
        #     Istart, Iend = as_backend_type(MTC).vec().getOwnershipRange()
        #     for I in arange(Istart, Iend):
        #         as_backend_type(MTC).vec()[I] \
        #             = self.getHomogeneousCoordinate(I,i)
        #     as_backend_type(MTC).vec().assemblyBegin()
        #     as_backend_type(MTC).vec().assemblyEnd()
        #
        #     self.cpFuncs[i].vector().set_local((self.M_control*MTC).get_local())
        #     as_backend_type(self.cpFuncs[i].vector()).vec().ghostUpdate()
        #
        # # may need to be permuted
        # self.zeroDofs = [] #self.generateZeroDofs()
        #
        # # replace M with permuted version
        # #if(mpisize > 1):
        # #
        # #    self.permutation = self.generatePermutation()
        # #    newM = self.M.mat()\
        # #                 .permute\
        # #                 (generateIdentityPermutation\
        # #                  (self.M.mat().getOwnershipRange()),\
        # #                  self.permutation)
        # #    self.M = PETScMatrix(newM)
        # #
        # #    # fix list of zero DOFs
        # #    self.permutationAO = PETSc.AO()
        # #    self.permutationAO\
        # #        .createBasic(self.permutation,\
        # #                     generateIdentityPermutation\
        # #                     (self.M.mat().getOwnershipRangeColumn()))
        # #    zeroDofIS = PETSc.IS()
        # #    zeroDofIS.createGeneral(array(self.zeroDofs,dtype=INDEX_TYPE))
        # #    self.zeroDofs = self.permutationAO.app2petsc\
        # #                    (zeroDofIS).getIndices()

    def apply_permutation(self):
        """
        Permutes the order of the IGA degrees of freedom, so that their
        parallel partitioning better aligns with that of the FE degrees 
        of freedom, which is generated by standard mesh-partitioning
        approaches in FEniCS.  
        """
        if dolfin.MPI.size(self.comm) > 1:
            self.permutation = self.generatePermutation()
            newM = self.M.mat().permute(
                generateIdentityPermutation(
                    self.M.mat().getOwnershipRange(), self.comm),
                self.permutation)
            self.M = PETScMatrix(newM)

            # fix list of zero DOFs
            self.permutationAO = PETSc.AO(self.comm)
            self.permutationAO \
                .createBasic(self.permutation,
                             generateIdentityPermutation
                             (self.M.mat().getOwnershipRangeColumn(),
                              self.comm))
            zeroDofIS = PETSc.IS(self.comm)
            zeroDofIS.createGeneral(
                numpy.array(self.zero_dofs, dtype=INDEX_TYPE))
            self.zero_dofs = self.permutationAO.app2petsc(
                zeroDofIS).getIndices()

    def write_extraction(self, dir_name: str,
                         do_permutation: bool = DEFAULT_DO_PERMUTATION):
        """
        Write extract data to disk:

        * HDF5 File
            - mesh
            - extracted control points and weights
        * Serialization of M_control
        * Serialization of M
        * Text metadata
            - Number of spatial dimensions (``nsd``)
            - Number of fields for each field and scalar control field
            - Function space metadata: element type and degree
        * File for each processor listing zeroed DoFs

        Parameters
        ----------
        dir_name : Output directory
        do_permutation : Permute DoFs for improved parallel matmult performance.
            Setting to True may be slow for large meshes
        """
        if do_permutation:
            self.apply_permutation()

        # write HDF file
        f = dolfin.HDF5File(self.comm,
                            dir_name + "/" + EXTRACTION_DATA_FILE, "w")
        f.write(self.mesh, EXTRACTION_H5_MESH_NAME)

        for i in range(0, self.nsd + 1):
            f.write(self.cpFuncs[i], EXTRACTION_H5_CONTROL_FUNC_NAME(i))
        f.close()

        # PETSc matrices
        viewer = PETSc.Viewer(self.comm) \
            .createBinary(dir_name + "/" + EXTRACTION_MAT_FILE, 'w')
        viewer(self.M.mat())
        viewer = PETSc.Viewer(self.comm) \
            .createBinary(dir_name + "/" + EXTRACTION_MAT_FILE_CTRL, 'w')
        viewer(self.M_control.mat())

        # write out zero-ed dofs
        zeroDofIS = PETSc.IS(self.comm)
        zeroDofIS.createGeneral(numpy.array(self.zero_dofs, dtype=INDEX_TYPE))
        viewer = PETSc.Viewer(self.comm) \
            .createBinary(dir_name + "/" + EXTRACTION_ZERO_DOFS_FILE, 'w')
        viewer(zeroDofIS)

        # write info
        if mpirank == 0:
            fs = str(self.nsd) + "\n" \
                 + self.extraction_element() + "\n" \
                 + str(self.get_num_fields()) + "\n"
            for i in range(-1, self.get_num_fields()):
                fs += str(self.get_degree(i)) + "\n" \
                      + str(self.get_ncp(i)) + "\n"
            f = open(dir_name + "/" + EXTRACTION_INFO_FILE, 'w')
            f.write(fs)
            f.close()
        dolfin.MPI.barrier(self.comm)


class ExtractedNonlinearProblem(dolfin.NonlinearProblem):
    """
    Class encapsulating a nonlinear problem posed on an extracted spline, to
    allow existing nonlinear solvers (e.g., PETSc SNES) to be used.

    NOTE: Obtaining the initial guess for the IGA DoFs from the given 
    FE function for the solution fields currently requires
    a linear solve, which is performed using the spline object's solver,
    if any.
    """
    def __init__(self,spline,residual,tangent,solution,**kwargs):
        """
        The argument ``spline`` is an ``ExtractedSpline`` on which the
        problem is solved.  ``residual`` is the residual form of the problem.
        ``tangent`` is the Jacobian of this form.  ``solution`` is a 
        ``Function`` in ``spline.V``.  Additional keyword arguments will be
        passed to the superclass constructor.
        """
        super(ExtractedNonlinearProblem, self).__init__(**kwargs)
        self.spline = spline
        self.solution = solution
        self.residual = residual
        self.tangent = tangent

    # Override methods from NonlinearProblem to perform extraction:
    def form(self,A,P,B,x):
        self.solution.vector()[:] = self.spline.M*x
    def F(self,b,x):
        b[:] = self.spline.assemble_vector(self.residual)
        return b
    def J(self,A,x):
        M = self.spline.assemble_matrix(self.tangent).mat()
        A.mat().setSizes(M.getSizes())
        A.mat().setUp()
        M.copy(result=A.mat())
        return A


class ExtractedNonlinearSolver:
    """
    Class encapsulating the extra work surrounding a nonlinear solve when
    the problem is posed on an ``ExtractedSpline``.
    """
    def __init__(self,problem,solver):
        """
        ``problem`` is an ``ExtractedNonlinearProblem``, while ``solver`` 
        is either a ``NewtonSolver`` or a ``PETScSNESSolver``
        that will be used behind the scenes.
        """
        self.problem = problem
        self.solver = solver

    def solve(self):
        """
        This method solves ``self.problem``, using ``self.solver`` and updating 
        ``self.problem.solution`` with the solution (in extracted FE 
        representation).
        """

        # Need to solve a linear problem for initial guess for IGA DoFs; any
        # way around this?
        tempVec = self.problem.spline.FEtoIGA(self.problem.solution)

        #tempFunc = Function(self.problem.spline.V)
        #tempFunc.assign(self.problem.solution)
        ## RHS of problem for initial guess IGA DoFs:
        #MTtemp = self.problem.spline.extractVector(tempFunc.vector(),
        #                                           applyBCs=False)
        ## Vector with right dimension for IGA DoFs (content doesn't matter):
        #tempVec = self.problem.spline.extractVector(tempFunc.vector())
        ## LHS of problem for initial guess:
        #Mm = as_backend_type(self.problem.spline.M).mat()
        #MTMm = Mm.transposeMatMult(Mm)
        #MTM = PETScMatrix(MTMm)
        #if(self.problem.spline.linearSolver == None):
        #    solve(MTM,tempVec,MTtemp)
        #else:
        #    self.problem.spline.linearSolver.solve(MTM,tempVec,MTtemp)
        self.solver.solve(self.problem,tempVec)

        self.problem.solution.vector()[:] = self.problem.spline.M*tempVec


#class SplineDisplacementExpression(Expression):
#
#    """
#    An expression that can be used to evaluate ``F`` plus an optional 
#    displacement at arbitrary points.  To be usable, it must have the 
#    following attributes assigned: 
#
#    (1) ``self.spline``: an instance of ``ExtractedSpline`` to which the 
#    displacement applies. 
#
#    (2) ``self.functionList:`` a list of scalar functions in the 
#    function space for ``spline``'s control mesh, which act as components of 
#    the displacement. If ``functionList`` contains too few entries (including 
#    zero entries), the missing entries are assumed to be zero.
#    """
#    
#    # needs attributes:
#    # - spline (ExtractedSpline)
#    # - functionList (list of SCALAR Functions)
#    
#    def eval_cell(self,values,x,c):
#        phi = []
#        out = array([0.0,])
#        for i in range(0,self.spline.nsd):
#            self.spline.cpFuncs[i].set_allow_extrapolation(True)
#            #phi += [self.cpFuncs[i](Point(x)),]
#            self.spline.cpFuncs[i].eval_cell(out,x,c)
#            phi += [out[0],]
#        self.spline.cpFuncs[self.spline.nsd].set_allow_extrapolation(True)
#        for i in range(0,self.spline.nsd):
#            if(i<len(self.functionList)):
#                self.functionList[i].set_allow_extrapolation(True)
#                self.functionList[i].eval_cell(out,x,c)
#                phi[i] += out[0]
#        #w = self.cpFuncs[self.nsd](Point(x))
#        self.spline.cpFuncs[self.spline.nsd].eval_cell(out,x,c)
#        w = out[0]
#        for i in range(0,self.spline.nsd):
#            phi[i] = phi[i]/w
#        xx = []
#        for i in range(0,self.spline.nsd):
#            if(i<len(x)):
#                xx += [x[i],]
#            else:
#                xx += [0,]
#        for i in range(0,self.spline.nsd):
#            values[i] = phi[i] - xx[i]
#            
#    #def value_shape(self):
#    #    return (self.spline.nsd,)


# compose with deformation
#class tIGArExpression(Expression):
#
#    """
#    A subclass of ``Expression`` which composes its attribute ``self.expr``
#    (also an ``Expression``) with the deformation ``F`` given by its attribute 
#    ``self.cpFuncs``, which is a list of ``Function`` objects, specifying the 
#    components of ``F``.
#    """
#
#    # using eval_cell allows us to avoid having to search for which cell
#    # x is in; also x need not be in a unique cell, which is nice for
#    # splines that do not have a single coordinate chart
#    def eval_cell(self,values,x,c):
#        phi = []
#        out = array([0.0,])
#        for i in range(0,self.nsd):
#            self.cpFuncs[i].set_allow_extrapolation(True)
#            self.cpFuncs[i].eval_cell(out,x,c)
#            phi += [out[0],]
#        self.cpFuncs[self.nsd].set_allow_extrapolation(True)
#        self.cpFuncs[self.nsd].eval_cell(out,x,c)
#        w = out[0]
#        for i in range(0,self.nsd):
#            phi[i] = phi[i]/w
#        self.expr.eval(values,array(phi))


class ExtractedSpline:
    """
    A class representing an extracted spline.  The idea is that all splines
    look the same after extraction, so there is no need for a proliferation
    of different classes to cover NURBS, T-splines, etc. (as there is for
    extraction generators).  
    """

    @functools.singledispatchmethod
    def __init__(self, arg):
        raise NotImplementedError(
            "Constructor not implemented for %s" % str(type(arg)))

    @__init__.register
    def _(self, generator: AbstractExtractionGenerator,
          quad_deg: int, do_permutation: bool=DEFAULT_DO_PERMUTATION):
        """
        Generates instance from an ``AbstractExtractionGenerator``, without
        passing through the filesystem.  This mainly exists to circumvent
        broken parallel HDF5 file output for quads and hexes in 2017.2
        (See Issue #1000 for DOLFIN on Bitbucket.)

        Notes
        -----
        While seemingly-convenient for small-scale testing/demos, and
        more robust in the sense that it makes no assumptions about the
        DoF ordering in FunctionSpaces being deterministic,
        this is not the preferred workflow for most realistic
        cases, as it forces a possibly-expensive preprocessing step to
        execute every time the analysis code is run.

        Parameters
        ----------
        generator: Spline extraction
        quad_deg: Quadrature degree using in integration of spline
        do_permutation: Choose whether or not to apply a permutation to the IGA
            DoF order
        """
        if do_permutation:
            generator.apply_permutation()

        self.quad_deg = quad_deg
        self.nsd = generator.get_nsd()
        self.element_type = generator.extraction_element()
        self.n_fields = generator.get_num_fields()
        self.p_control = generator.get_degree(-1)
        self.p = [generator.get_degree(i) for i in range(0, self.n_fields)]
        self.mesh = generator.mesh
        self.cp_funcs = generator.cpFuncs
        self.VE = generator.VE
        self.VE_control = generator.VE_control
        self.V = generator.V
        self.V_control = generator.V_control
        self.M = generator.M
        self.M_control = generator.M_control
        self.comm = generator.getComm()
        zero_dof_is = PETSc.IS(self.comm)
        zero_dof_is.createGeneral(
            numpy.array(generator.zero_dofs, dtype=INDEX_TYPE))
        self.zero_dofs = zero_dof_is

        self.generic_setup()

    @__init__.register
    def _(self, dir_name: str, quad_deg: int, mesh: dolfin.Mesh,
          comm=dolfin.MPI.comm_world):
        """
        Generates instance from extraction data in the provided directory.
        Optionally takes a DOLFIN mesh argument, so that function spaces can be
        established on the same mesh as an existing spline object for
        facilitating segregated solver schemes. (Splines common to one set of
        extraction data are always treated as a monolothic mixed function
        space). Everything to do with the spline is integrated using a
        quadrature rule implementing the provided degree.

        Parameters
        ----------
        dir_name: Corresponding to directory containing spline extraction data
        quad_deg: Quadrature degree using in integration of spline
        mesh: Function spaces can be established on the same mesh as an
            existing spline object for facilitating segregated solver schemes.
            (Splines common to one set of extraction data are always treated
            as a monolothic mixed function space.)
        comm: MPI communicator
        """
        self.quad_deg = quad_deg
        self.comm = comm

        # read function space info
        f = open(dir_name + "/" + EXTRACTION_INFO_FILE, 'r')
        fs = f.read()
        f.close()
        lines = fs.split('\n')
        lineCount = 0
        self.nsd = int(lines[lineCount])
        lineCount += 1
        self.element_type = lines[lineCount]
        lineCount += 1
        self.n_fields = int(lines[lineCount])
        lineCount += 1
        self.p_control = int(lines[lineCount])
        lineCount += 1
        ncp_control = int(lines[lineCount])
        lineCount += 1
        self.p = []
        ncp = []
        for i in range(0, self.n_fields):
            self.p += [int(lines[lineCount]), ]
            lineCount += 1
            ncp += [int(lines[lineCount]), ]
            lineCount += 1
        # prealloc_control = int(lines[lineCount])
        # lineCount += 1
        # prealloc = int(lines[lineCount])

        # read mesh if none provided
        # f = HDF5File(mpi_comm_world(),dirname+"/"+EXTRACTION_DATA_FILE,'r')
        f = dolfin.HDF5File(self.comm, dir_name + "/" + EXTRACTION_DATA_FILE, 'r')
        if (mesh == None):
            self.mesh = dolfin.Mesh(self.comm)

            # NOTE: behaves erratically in parallel for quad/hex meshes
            # in 2017.2; hopefully will be fixed soon (see dolfin
            # issue #1000).
            f.read(self.mesh, EXTRACTION_H5_MESH_NAME, True)

        else:
            self.mesh = mesh

        # create function spaces
        self.VE_control \
            = ufl.FiniteElement(self.element_type, self.mesh.ufl_cell(),
                            self.p_control)
        self.V_control \
            = dolfin.FunctionSpace(self.mesh, self.VE_control)

        if self.n_fields > 1:
            VE_components = []
            for i in range(0, self.n_fields):
                VE_components \
                    += [ufl.FiniteElement(self.element_type,
                                          self.mesh.ufl_cell(), self.p[i]), ]
            self.VE = ufl.MixedElement(tuple(VE_components))
        else:
            self.VE = ufl.FiniteElement(
                self.element_type, self.mesh.ufl_cell(), self.p[0])

        self.V = dolfin.FunctionSpace(self.mesh, self.VE)

        # read control functions
        self.cp_funcs = []
        for i in range(0, self.nsd + 1):
            self.cp_funcs += [dolfin.Function(self.V_control), ]
            f.read(self.cp_funcs[i],
                   EXTRACTION_H5_CONTROL_FUNC_NAME(i))
        f.close()

        # read extraction matrix and create transpose for control space
        Istart, Iend = dolfin.as_backend_type(
            self.cp_funcs[0].vector()).vec().getOwnershipRange()
        nLocalNodes = Iend - Istart
        MPETSc = PETSc.Mat(self.comm)
        MPETSc.create(PETSc.COMM_WORLD)
        # arguments: [[localRows,localColumns],[globalRows,globalColums]]
        # or is it [[localRows,globalRows],[localColumns,globalColums]]?
        # the latter seems to be what comes out of getSizes()...
        if (dolfin.MPI.size(self.comm) > 1):
            MPETSc.setSizes([[nLocalNodes, None], [None, ncp_control]])
        # MPETSc.setType('aij') # sparse
        # MPETSc.setPreallocationNNZ(prealloc_control)
        # MPETSc.setUp()
        viewer = PETSc.Viewer(self.comm).createBinary(
            dir_name + "/" + EXTRACTION_MAT_FILE_CTRL, 'r')

        self.M_control = dolfin.PETScMatrix(MPETSc.load(viewer))

        # read extraction matrix and create transpose
        Istart, Iend = dolfin.as_backend_type(
            dolfin.Function(self.V).vector()).vec().getOwnershipRange()
        nLocalNodes = Iend - Istart
        totalDofs = 0
        for i in range(0, self.n_fields):
            totalDofs += ncp[i]
        MPETSc2 = PETSc.Mat(self.comm)
        MPETSc2.create(PETSc.COMM_WORLD)
        # arguments: [[localRows,localColumns],[globalRows,globalColums]]
        if dolfin.MPI.size(self.comm) > 1:
            MPETSc2.setSizes([[nLocalNodes, None], [None, totalDofs]])
        # MPETSc2.setType('aij') # sparse
        # MPETSc2.setPreallocationNNZ(prealloc)
        # MPETSc2.setUp()
        viewer = PETSc.Viewer(self.comm).createBinary(
            dir_name + "/" + EXTRACTION_MAT_FILE, 'r')
        self.M = dolfin.PETScMatrix(MPETSc2.load(viewer))

        # read zero-ed dofs
        # f = open(dirname+"/"+EXTRACTION_ZERO_DOFS_FILE(mpirank),"r")
        # f = open(dirname+"/"+EXTRACTION_ZERO_DOFS_FILE,"r")
        # fs = f.read()
        # f.close()
        # dofStrs = fs.split()
        # zeroDofs  = []
        # for dofStr in dofStrs:
        #    # only keep the ones for this processor
        #    possibleDof = int(dofStr)
        #    if(possibleDof < Iend and possibleDof >= Istart):
        #        zeroDofs += [possibleDof,]
        # self.zeroDofs = PETSc.IS()
        # self.zeroDofs.createGeneral(array(zeroDofs,dtype=INDEX_TYPE))

        viewer = PETSc.Viewer(self.comm).createBinary(
            dir_name + "/" + EXTRACTION_ZERO_DOFS_FILE, "r")
        self.zero_dofs = PETSc.IS(self.comm)
        self.zero_dofs.load(viewer)
        self.generic_setup()

    def generic_setup(self):
        """
        Setup steps to take regardless of the source of extraction data.
        """
        # for marking subdomains
        # self.boundaryMarkers = FacetFunctionSizet(self.mesh,0)
        # self.boundaryMarkers \
        #    = MeshFunctionSizet(self.mesh,self.mesh.topology().dim()-1,0)
        self.boundaryMarkers = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1, 0)

        # caching transposes of extraction matrices
        if FORM_MT:
            self.MT_control = dolfin.PETScMatrix(
                self.M_control.mat().transpose(PETSc.Mat(self.comm)))
            self.MT = dolfin.PETScMatrix(
                self.M.mat().transpose(PETSc.Mat(self.comm)))

        # geometrical mapping
        components = []
        for i in range(0, self.nsd):
            components += [self.cp_funcs[i] / self.cp_funcs[self.nsd], ]
        self.F = ufl.as_vector(components)
        self.DF = grad(self.F)

        # debug
        # self.DF = Identity(self.nsd)

        # metric tensor
        self.g = getMetric(self.F)  # (self.DF.T)*self.DF

        # normal of pre-image in coordinate chart
        self.N = ufl.FacetNormal(self.mesh)

        # normal that preserves orthogonality w/ pushed-forward tangent vectors
        self.n = mappedNormal(self.N, self.F)

        # integration measures
        self.dx = ScaledMeasure(volumeJacobian(self.g), dolfin.dx,
                                self.quad_deg)
        self.ds = ScaledMeasure(surfaceJacobian(self.g, self.N),
                                dolfin.ds, self.quad_deg, self.boundaryMarkers)

        # useful for defining Cartesian differential operators
        self.pinvDF = pinvD(self.F)

        # useful for tensors given in parametric coordinates
        self.gamma = getChristoffel(self.g)

        # linear space on mesh for projecting scalar fields onto
        self.VE_linear = ufl.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)

        # linearList = []
        # for i in range(0,self.nsd):
        #    linearList += [self.VE_linear,]
        # self.VE_displacement = MixedElement(linearList)

        self.VE_displacement = ufl.VectorElement(
            "Lagrange", self.mesh.ufl_cell(), 1, dim=self.nsd)

        # self.VE_displacement = VectorElement\
        #                       ("Lagrange",self.mesh.ufl_cell(),1)

        self.V_displacement = dolfin.FunctionSpace(
            self.mesh, self.VE_displacement)
        self.V_linear = dolfin.FunctionSpace(self.mesh, self.VE_linear)

        # TODO: customise linear solver
        self.linear_solver = None

    def FEtoIGA(self,u):
        """
        This solves the pseudoinverse problem to get the IGA degrees of
        freedom from the finite element ones associated with ``u``, which
        is a ``Function``.  It uses the ``self`` instance's linear solver
        object if available.  The return value is a ``PETScVector`` of the
        IGA degrees of freedom.

        NOTE: This is inefficient and should rarely be necessary.  It is 
        mainly intended for testing purposes, or as a last resort.
        """
        tempFunc = dolfin.Function(self.V)
        tempFunc.assign(u)
        # RHS of problem for initial guess IGA DoFs:
        MTtemp = self.extract_vector(tempFunc.vector(), apply_bcs=False)
        # Vector with right dimension for IGA DoFs (content doesn't matter):
        tempVec = self.extract_vector(tempFunc.vector())
        # LHS of problem for initial guess:
        Mm = dolfin.as_backend_type(self.M).mat()
        MTMm = Mm.transposeMatMult(Mm)
        MTM = dolfin.PETScMatrix(MTMm)
        if(self.linear_solver == None):
            dolfin.solve(MTM,tempVec,MTtemp)
        else:
            self.linear_solver.solve(MTM, tempVec, MTtemp)
        return tempVec

    #def interpolateAsDisplacement(self,functionList=[]):
    #
    #    """
    #    Given a list of scalar functions, get a displacement field from 
    #    mesh coordinates to control + functions in physical space, 
    #    interpolated on linear elements for plotting without discontinuities 
    #    on cut-up meshes. Default argument of ``functionList=[]`` 
    #    just interpolates the control functions.  If there are fewer elements
    #    in ``functionList`` than there are control functions, then the missing
    #    functions are assumed to be zero.
    #
    #    NOTE: Currently only works with extraction to simplicial elements.
    #    """
    #    
    #    #expr = SplineDisplacementExpression(degree=self.quadDeg)
    #    expr = SplineDisplacementExpression\
    #           (element=self.VE_displacement)
    #    expr.spline = self
    #    expr.functionList = functionList
    #    disp = Function(self.V_displacement)
    #    disp.interpolate(expr)
    #    return disp

    def grad(self, f, F=None):
        """
        Cartesian gradient in deformed configuration

        Parameters
        ----------
        f : Mathematical expression
        F : Optional explicitly provided mapping from parametric to physical
            space

        Returns
        -------
        Cartesian gradient with respect to physical coordinates

        Notes
        -----
        When applied to tensor-valued expressions, f is considered to be in
        the Cartesian coordinates of the physical configuration, *not* in the
        local coordinate chart with respect to which derivatives are taken by
        FEniCS.
        """
        if F is None:
            F = self.F
        return cartesian_grad(f, F)

    def div(self, f, F=None):
        """
        Parameters
        ----------
        f : Mathematical expression
        F : Optional explicitly provided mapping from parametric to physical
            space

        Returns
        -------
        Cartesian divergence with respect to physical coordinates
        """
        if F is None:
            F = self.F
        return cartesian_div(f, F)

    def curl(self, f, F=None):
        """
        Parameters
        ----------
        f : Mathematical expression
        F : Optional explicitly provided mapping from parametric to physical
            space

        Returns
        -------
        Cartesian curl with respect to physical coordinates

        Notes
        -----
        Only applies in 3D to vector-valued expressions.
        """
        if F is None:
            F = self.F
        return cartesian_curl(f, F)

    def parametric_grad(self, f):
        """
        Parameters
        ----------
        f : Mathematical expression

        Returns
        -------
        Cartesian gradient with respect to parametric coordinates
        """
        return grad(f)

    # curvilinear variants; if f is only a regular tensor, will create a
    # CurvilinearTensor w/ all indices lowered.  Metric defaults to one
    # generated by mapping self.F (into Cartesian space) if no metric is
    # supplied via f.
    def curvilinear_grad(self, f):
        """
        Covariant derivative of a curvilinear tensor  expression taken with
        respect to parametric coordinates, assuming  that the components of
        the expression are also given in this coordinate system.

        Notes
        -----
        If a regular tensor is passed as the argument, it will be converted
        to a ``CurvilinearTensor`` with all lowered indices.

        Parameters
        ----------
        f : Cuvilinear tensor

        Returns
        -------
        Curvilinear gradient
        """
        if not isinstance(f, CurvilinearTensor):
            f = CurvilinearTensor(f, self.g)
        return curvilinear_grad(f)

    def curvilinear_div(self, f):
        """
        Curvilinear divergence operator corresponding to reference
        configuration gradient. Contracts new lowered index from reference
        configuration gradient with last raised index of the provided argument.

        Notes
        -----
        If a regular tensor is passed as the argument, it will be converted
        to a ``CurvilinearTensor`` with all lowered indices.
        """
        if not isinstance(f, CurvilinearTensor):
            f = CurvilinearTensor(f, self.g).sharp()
        return curvilinear_div(f)

    #def spatialExpression(self,expr):
    #    """
    #    Converts string ``expr`` into an ``Expression``, 
    #    treating the coordinates ``'x[i]'`` in ``expr`` as 
    #    spatial coordinates.  
    #    (Using the standard ``Expression`` constructor, these would be treated 
    #    as parametric coordinates.)
    #
    #    NOTE: Only works when extracting to simplicial elements.
    #    """
    #    retval = tIGArExpression(degree=self.quadDeg)
    #    retval.expr = Expression(expr,degree=self.quadDeg)
    #    retval.nsd = self.nsd
    #    retval.cpFuncs = self.cpFuncs
    #    return retval

    def parametric_expression(self, expr) -> dolfin.Expression:
        """
        Create an ``Expression`` from a string, ``expr``, interpreting the
        coordinates ``'x[i]'`` in ``expr`` as parametric coordinates.
        Uses quadrature degree of spline object for interpolation degree.
        """
        return dolfin.Expression(expr, degree=self.quad_deg)

    def parametric_coordinates(self) -> ufl.SpatialCoordinate:
        """
        Wrapper for ``SpatialCoordiantes()`` to avoid confusion, since
        FEniCS's spatial coordinates are used in tIGAr as parametric 
        coordinates.  
        """
        return ufl.SpatialCoordinate(self.mesh)

    def spatial_coordinates(self):
        """
        Returns
        -------
        The mapping F which gives the spatial coordinates of a parametric point
        """
        return self.F

    def rationalize(self, u: typing.Any):
        """
        Divides its argument by the weighting function of the spline's
        control mesh.
        """
        return u/(self.cp_funcs[self.nsd])

    # split out to implement contact
    def extract_vector(self, b: dolfin.PETScVector, apply_bcs: bool = True):
        """
        Apply extraction to an FE vector. Optional boolean argument indicates
        whether or not to apply BCs to the vector.

        Parameters
        ----------
        b : PETScVector of coefficients
        apply_bcs : Apply BCs to resulting vector

        Returns
        -------
        Extracted vector M^T b
        """
        # MT determines parallel partitioning of MTb
        if FORM_MT:
            MTb = self.MT*b
        else:
            MTb = multTranspose(self.M, b)

        # apply zero bcs to MTAM and MTb
        if apply_bcs:
            dolfin.as_backend_type(MTb).vec().setValues(
                self.zero_dofs, numpy.zeros(self.zero_dofs.getLocalSize()))
            dolfin.as_backend_type(MTb).vec().assemblyBegin()
            dolfin.as_backend_type(MTb).vec().assemblyEnd()

        return MTb

    def assemble_vector(self, form: dolfin.Form, apply_bcs: bool = True):
        """
        Assemble M^T b where b is a vector assembled from the provided linear
        form and the optional argument indicates whether or not to apply the
        Dirichlet BCs.

        Parameters
        ----------
        form : Linear finite element form to assemble into b
        apply_bcs : Optional demand to apply BCs after assembly

        Returns
        -------
        M^T b
        """
        b = dolfin.assemble(form)
        MTb = self.extract_vector(b, apply_bcs=apply_bcs)
        return MTb

    def extract_matrix(self, A: dolfin.PETScMatrix, apply_bcs: bool = True,
                       diag: int = 1) -> dolfin.PETScMatrix:
        """
        Apply extraction to an FE matrix.

        Parameters
        ----------
        A : Finite element matrix
        apply_bcs : Optional argument indicating whether to apply Dirichlet
            BCs
        diag : Where Dirichlet BCs are applied, this value is inserted into
            the corresponding diagonal entries

        Returns
        -------
        The extracted matrix M^T A M
        """
        if FORM_MT:
            Am = dolfin.as_backend_type(A).mat()
            MTm = dolfin.as_backend_type(self.MT).mat()
            MTAm = MTm.matMult(Am)
            Mm = dolfin.as_backend_type(self.M).mat()
            MTAMm = MTAm.matMult(Mm)
            MTAM = dolfin.PETScMatrix(MTAMm)
        else:
            # Needs recent version of petsc4py; seems to be standard version
            # used in docker container, though, since this function works
            # fine on stampede.
            MTAM = dolfin.PETScMatrix(
                dolfin.as_backend_type(A).mat().PtAP(
                    dolfin.as_backend_type(self.M).mat()))

        # apply zero bcs to MTAM and MTb
        # (default behavior is to set diag=1, as desired)
        if apply_bcs:
            dolfin.as_backend_type(MTAM).mat().zeroRowsColumns(
                self.zero_dofs, diag)
        dolfin.as_backend_type(MTAM).mat().assemblyBegin()
        dolfin.as_backend_type(MTAM).mat().assemblyEnd()

        return MTAM

    def assemble_matrix(self, form, applyBCs=True, diag=1) \
            -> dolfin.PETScMatrix:
        """
        Assemble M^T*A*M where A is the finite element matrix corresponding
        to the finite element form argument.

        Parameters
        ----------
        form : Finite element formulation to be assembled
        apply_bcs : Optional argument indicating whether to apply Dirichlet
            BCs
        diag : Where Dirichlet BCs are applied, this value is inserted into
            the corresponding diagonal entries

        Returns
        -------
        The assembled and extracted matrix M^T A M
        """
        A = dolfin.PETScMatrix(self.comm)
        dolfin.assemble(form, tensor=A)
        MTAM = self.extract_matrix(A, apply_bcs=applyBCs, diag=diag)
        return MTAM

    def assemble_linear_system(self, a: ufl.Form, L: ufl.Form,
                               apply_bcs: bool = True) \
            -> typing.Tuple[dolfin.PETScMatrix, dolfin.PETScVector]:
        """
        Parameters
        ----------
        a : Bilinear formulation
        L : Linear formulation
        apply_bcs : Indicate whether BCs should be applied

        Returns
        -------
        The finite element matrix and vector
        """
        A = self.assemble_matrix(a, apply_bcs)
        b = self.assemble_vector(L, apply_bcs)
        return A, b

    def solve_linear_system(self, MTAM: dolfin.PETScMatrix,
                            MTb: dolfin.PETScVector,
                            u: dolfin.Function) -> dolfin.PETScVector:
        """
        Solves M^T A M U = M^T b where U is the vector of IGA unknowns (in
        the homogeneous coordinate representation if rational splines are
        being used).

        Parameters
        ----------
        MTAM : The IGA matrix operator
        MTb : The IGA right hand side vector
        u : Solution vector of FE unknowns

        Returns
        -------
        M^T U
        """
        U = u.vector()
        if FORM_MT:
            MTU = self.MT * U
        else:
            MTU = multTranspose(self.M, U)
        if self.linear_solver is None:
            dolfin.solve(MTAM, MTU, MTb)
        else:
            self.linear_solver.solve(MTAM, MTU, MTb)
        u.vector().set_local((self.M * MTU).get_local())
        dolfin.as_backend_type(u.vector()).vec().ghostUpdate()
        dolfin.as_backend_type(u.vector()).vec().assemble()

        return MTU

    def solve_linear_variational_problem(
            self, residual: typing.Union[ufl.Form, ufl.equation.Equation],
            u: dolfin.Function, apply_bcs: bool=True) -> dolfin.PETScVector:
        """
        Solves a linear variational problem defined in its residual
        formulation. Homogeneous Dirichlet BCs are optionally applied. The
        return value of the function is the vector of IGA degrees of freedom.

        Parameters
        ----------
        residual : Finite element residual formulation
        u : Solution function
        apply_bcs : Indicates whether to apply Dirichlet BCs

        Returns
        -------
        Vector of IGA degreees of freedom M^T U
        """
        if isinstance(residual, ufl.equation.Equation):
            lhs_form = residual.lhs
            rhs_form = residual.rhs
        else:
            # TODO: Why is this so much slower?
            lhs_form = ufl.lhs(residual)
            rhs_form = ufl.rhs(residual)

        if rhs_form.integrals() == ():
            v = ufl.TestFunction(self.V)
            rhs_form = dolfin.Constant(0.0)*v[0]*self.dx

        MTAM, MTb = self.assemble_linear_system(lhs_form, rhs_form, apply_bcs)
        return self.solve_linear_system(MTAM, MTb, u)

    def solve_nonlinear_variational_problem(
            self, residual_form: ufl.Form, J: ufl.Form, u: dolfin.Function,
            reference_error=None, iga_dofs=None, max_iters: int = 20,
            relative_tolerance: float = 1e-5):
        """
        Solves the nonlinear variational problem defined by the input arguments.

        Parameters
        ----------
        residual_form : Residual formulation
        J : Jacobian of the residual formulation
        u : Solution function
        reference_error : Optionally provided for computation of the relative
            error
        iga_dofs : Initial guess vector of IGA DoFs
        max_iters : Maximum number of Newton iterations
        relative_tolerance : Relative error tolerance

        Warnings
        --------
        If ``iga_dofs`` is provided, the entries in ``u``'s vector will be
        overwritten for the initial guess. These IGA DoFs will also be
        overwritten by the IGA DoFs of the problem's solution.
        """
        # returning_dofs = not isinstance(iga_dofs, type(None))
        returning_dofs = iga_dofs is not None
        if returning_dofs:
            # Overwrite content of u with extraction of igaDoFs.
            u.vector().set_local((self.M * iga_dofs).get_local())
            dolfin.as_backend_type(u.vector()).vec().ghostUpdate()
            dolfin.as_backend_type(u.vector()).vec().assemble()

        # Newton iteration loop:
        converged = False
        for i in range(0, max_iters):
            MTAM, MTb = self.assemble_linear_system(J, residual_form)
            current_norm = dolfin.norm(MTb)
            if i == 0 and reference_error is None:
                reference_error = current_norm
            relative_norm = current_norm / reference_error
            if dolfin.MPI.rank(self.comm) == 0:
                print("Solver iteration: " + str(i)
                      + " , Relative norm: " + str(relative_norm))
                sys.stdout.flush()
            if relative_norm < relative_tolerance:
                converged = True
                break
            du = dolfin.Function(self.V)
            iga_increment = self.solve_linear_system(MTAM, MTb, du)
            u.assign(u - du)
            if returning_dofs:
                iga_dofs -= iga_increment
        if not converged:
            raise Exception("Nonlinear solver failed to converge")

    def project_scalar_onto_linears(
            self, f, linear_solver: dolfin.PETScLUSolver = None,
            lump_mass: bool = False) -> dolfin.Function:
        """
        Computes the L2 projection of an expression to a linear and scalar
        finite element function space. This method is typically useful for
        visualization and plotting.

        Parameters
        ----------
        f : Expression to project
        linear_solver : Linear solver to be employed
        lump_mass : Indicate whether mass lumping be used

        Returns
        -------
        The computed projection
        """
        u = ufl.TrialFunction(self.V_linear)
        v = ufl.TestFunction(self.V_linear)
        # don't bother w/ change of variables in integral
        #res = inner(u-toProject,v)*self.dx.meas

        # Note: for unclear reasons, extracting the lhs/rhs from the
        # residual is both itself very slow, and also causes the assembly
        # to become very slow.  

        if lump_mass:
            lhs_form = inner(dolfin.Constant(1.0), v) * self.dx.meas
        else:
            lhs_form = inner(u, v) * self.dx.meas  # lhs(res)
        rhs_form = inner(f, v) * self.dx.meas  # rhs(res)
        A = dolfin.assemble(lhs_form)
        b = dolfin.assemble(rhs_form)
        u = dolfin.Function(self.V_linear)
        # solve(A,u.vector(),b)
        if lump_mass:
            dolfin.as_backend_type(u.vector()) \
                .vec().pointwiseDivide(dolfin.as_backend_type(b).vec(),
                                       dolfin.as_backend_type(A).vec())
        else:
            if linear_solver is None:
                dolfin.solve(A, u.vector(), b)
            else:
                linear_solver.solve(A, u.vector(), b)
        return u

    def project(self, f, apply_bcs: bool = False, rationalize: bool = True,
                lump_mass: bool = False):
        """
        Computes the L2 projection of an expression to the extracted spline's
        solution space.

        Parameters
        ----------
        f : Expression to project
        apply_bcs : Indicate whether Dirichlet BCs should be applied
        rationalize : Indicate whether the returned value should be
            symbolically (by UFL) scaled by the weights
        lump_mass : Indicate whether to use mass lumping

        Returns
        -------
        The projected expression
        """
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        u = self.rationalize(u)
        v = self.rationalize(v)
        rhs_form = inner(f, v) * self.dx
        retval = dolfin.Function(self.V)
        if not lump_mass:
            lhs_form = inner(u, v) * self.dx
            self.solve_linear_variational_problem(lhs_form == rhs_form,
                                                  retval, apply_bcs)
        else:
            if self.n_fields == 1:
                one_constant = dolfin.Constant(1.0)
            else:
                one_constant = dolfin.Constant(self.n_fields * (1.0,))
            lhs_form = inner(one_constant, v) * self.dx
            lhs_vec_fe = dolfin.assemble(lhs_form)
            lhs_vec = self.extract_vector(lhs_vec_fe, apply_bcs=False)
            rhs_vec_fe = dolfin.assemble(rhs_form)
            rhs_vec = self.extract_vector(rhs_vec_fe, apply_bcs=apply_bcs)
            iga_dofs = self.extract_vector(dolfin.Function(self.V).vector())
            dolfin.as_backend_type(iga_dofs).vec() \
                .pointwiseDivide(dolfin.as_backend_type(rhs_vec).vec(),
                                 dolfin.as_backend_type(lhs_vec).vec())
            retval.vector()[:] = self.M * iga_dofs
        if rationalize:
            retval = self.rationalize(retval)
        return retval


class AbstractCoordinateChartSpline(AbstractExtractionGenerator):

    """
    This abstraction represents a spline whose parametric 
    coordinate system consists of a 
    using a single coordinate chart, so coordinates provide a unique set 
    of basis functions; this applies to single-patch B-splines, T-splines, 
    NURBS, etc., and, with a little creativity, can be stretched to cover
    multi-patch constructions.
    """

    @abc.abstractmethod
    def getNodesAndEvals(self,x,field):
        """
        Given a parametric point ``x``, return a list of the form
        
        ``[[index0, N_index0(x)], [index1,N_index1(x)], ... ]``
        
        where ``N_i`` is the ``i``-th basis function of the scalar polynomial 
        spline space (NOT of the rational space) corresponding to a given
        ``field``.
        """
        pass

    # return a matrix M for extraction
    def generateM_control(self):
        """
        Generates the extraction matrix for the single scalar spline space
        used to represent all homogeneous components of the mapping ``F``
        from parametric to physical space.
        """

        func = dolfin.Function(self.V_control)
        Istart, Iend = dolfin.as_backend_type(
            func.vector()).vec().getOwnershipRange()

        nLocalNodes = Iend - Istart
        x_nodes = self.V_control.tabulate_dof_coordinates().reshape(
            (-1, self.mesh.geometry().dim()))

        MPETSc = PETSc.Mat(self.comm)

        #MPETSc.create(PETSc.COMM_WORLD)
        # arguments: [[localRows,localColumns],[globalRows,globalColums]]
        #MPETSc.setSizes([[nLocalNodes,None],[None,self.getNcp(-1)]])
        #MPETSc.setType('aij') # sparse

        #MPETSc.create()

        MPETSc.createAIJ([[nLocalNodes,None], [None, self.get_ncp(-1)]],
                         comm=self.comm)
        MPETSc.setPreallocationNNZ([self.get_prealloc(True),
                                    self.get_prealloc(True)])

        # just slow down quietly if preallocation is insufficient
        MPETSc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        # for debug:
        #MPETSc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        MPETSc.setUp()

        # I indexes FEM nodes owned by this process
        #for I in xrange(Istart, Iend):
        dofs = self.V_control.dofmap().dofs()
        for I in numpy.arange(0,len(dofs)):
            x = x_nodes[dofs[I]-Istart]
            matRow = dofs[I]
            nodesAndEvals = self.getNodesAndEvals(x,-1)

            #cols = array(nodesAndEvals,dtype=INDEX_TYPE)[:,0]
            #rows = array([matRow,],dtype=INDEX_TYPE)
            #values = npTranspose(array(nodesAndEvals)[:,1:2])
            #MPETSc.setValues(rows,cols,values,addv=PETSc.InsertMode.INSERT)

            for i in range(0,len(nodesAndEvals)):
                if(abs(nodesAndEvals[i][1]) > self.getIgnoreEps()):
                    MPETSc[matRow,nodesAndEvals[i][0]] = nodesAndEvals[i][1]

        MPETSc.assemblyBegin()
        MPETSc.assemblyEnd()

        return dolfin.PETScMatrix(MPETSc)

    def generateM(self):
        """
        Generates the extraction matrix for the mixed function space of
        all unkown scalar fields.
        """

        func = dolfin.Function(self.V)
        Istart, Iend = dolfin.as_backend_type(
            func.vector()).vec().getOwnershipRange()
        nLocalNodes = Iend - Istart

        totalDofs = 0
        for i in range(0, self.get_num_fields()):
            totalDofs += self.get_ncp(i)

        MPETSc = PETSc.Mat(self.comm)
        #MPETSc.create(PETSc.COMM_WORLD)
        # arguments: [[localRows,localColumns],[globalRows,globalColums]]
        #MPETSc.setSizes([[nLocalNodes,None],[None,totalDofs]])
        MPETSc.createAIJ([[nLocalNodes,None],[None,totalDofs]],comm=self.comm)
        #MPETSc.setType('aij') # sparse
        # TODO: maybe change preallocation stuff
        MPETSc.setPreallocationNNZ([self.get_prealloc(False),
                                    self.get_prealloc(False)])
        #MPETSc.setPreallocationNNZ(0)
        # just slow down quietly if preallocation is insufficient
        MPETSc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        # for debug:
        #MPETSc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        MPETSc.setUp()

        offset = 0
        for field in range(0, self.get_num_fields()):
            x_nodes = self.V.tabulate_dof_coordinates()\
                            .reshape((-1,self.mesh.geometry().dim()))
            if(self.get_num_fields()>1):
                dofs = self.V.sub(field).dofmap().dofs()
            else:
                dofs = self.V.dofmap().dofs()
            for I in numpy.arange(0,len(dofs)):
                x = x_nodes[dofs[I]-Istart]
                matRow = dofs[I]
                nodesAndEvals = self.getNodesAndEvals(x,field)

                # Ideally, would use globalDof here for consistency,
                # but it is not very efficient as implemented
                #cols = array(nodesAndEvals,dtype=INDEX_TYPE)[:,0] + offset
                #rows = array([matRow,],dtype=INDEX_TYPE)
                #values = npTranspose(array(nodesAndEvals)[:,1:2])
                #MPETSc.setValues(rows,cols,values,addv=PETSc.InsertMode.INSERT)

                for i in range(0,len(nodesAndEvals)):
                    # Ideally, would use globalDof here for consistency,
                    # but it is not very efficient as implemented
                    if(abs(nodesAndEvals[i][1]) > self.getIgnoreEps()):
                        MPETSc[matRow,nodesAndEvals[i][0]+offset]\
                            = nodesAndEvals[i][1]

            offset += self.get_ncp(field)

        MPETSc.assemblyBegin()
        MPETSc.assemblyEnd()

        return dolfin.PETScMatrix(MPETSc)

    # override default behavior to order unknowns according to what task's
    # FE mesh they overlap.  this will (hopefully) reduce communication
    # cost in the matrix--matrix multiplies
    def generatePermutation(self):

        """
        Generates a permutation of the IGA degrees of freedom that tries to
        ensure overlap of their parallel partitioning with that of the FE
        degrees of freedom, which are partitioned automatically based on the
        FE mesh.
        """

        func = dolfin.Function(self.V)
        Istart, Iend = dolfin.as_backend_type(
            func.vector()).vec().getOwnershipRange()
        nLocalNodes = Iend - Istart

        totalDofs = 0
        for i in range(0, self.get_num_fields()):
            totalDofs += self.get_ncp(i)

        MPETSc = PETSc.Mat(self.comm)
        MPETSc.createAIJ([[nLocalNodes,None],[None,totalDofs]],comm=self.comm)
        MPETSc.setPreallocationNNZ([self.get_prealloc(False),
                                    self.get_prealloc(False)])
        MPETSc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        MPETSc.setUp()

        offset = 0
        for field in range(0, self.get_num_fields()):
            x_nodes = self.V.tabulate_dof_coordinates()\
                            .reshape((-1,self.mesh.geometry().dim()))
            if(self.get_num_fields()>1):
                dofs = self.V.sub(field).dofmap().dofs()
            else:
                dofs = self.V.dofmap().dofs()
            for I in numpy.arange(0,len(dofs)):
                x = x_nodes[dofs[I]-Istart]
                matRow = dofs[I]
                nodesAndEvals = self.getNodesAndEvals(x,field)

                #cols = array(nodesAndEvals,dtype=INDEX_TYPE)[:,0] + offset
                #rows = array([matRow,],dtype=INDEX_TYPE)
                #values = full((1,len(nodesAndEvals)),mpirank+1)
                #MPETSc.setValues(rows,cols,values,addv=PETSc.InsertMode.INSERT)

                for i in range(0,len(nodesAndEvals)):
                    MPETSc[matRow,nodesAndEvals[i][0]+offset]\
                        = mpirank+1 # need to avoid losing zeros...

            offset += self.get_ncp(field)

        MPETSc.assemblyBegin()
        MPETSc.assemblyEnd()

        MT = MPETSc.transpose(PETSc.Mat(self.comm))
        Istart, Iend = MT.getOwnershipRange()
        nLocal = Iend - Istart
        partitionInts = numpy.zeros(nLocal,dtype=INDEX_TYPE)
        for i in numpy.arange(Istart,Iend):
            rowValues = MT.getRow(i)[0]
            # isolate nonzero entries
            rowValues = numpy.extract(rowValues>0,rowValues)
            iLocal = i - Istart
            modeValues = scipy.stats.mode(rowValues)[0]
            if(len(modeValues) > 0):
                partitionInts[iLocal] = int(scipy.stats.mode(rowValues).mode[0]-0.5)
            else:
                partitionInts[iLocal] = 0 # necessary?
        partitionIS = PETSc.IS(self.comm)
        partitionIS.createGeneral(partitionInts,comm=self.comm)

        # kludgy, non-scalable solution:

        # all-gather the partition indices and apply argsort to their
        # underlying arrays
        bigIndices = numpy.argsort(
            partitionIS.allGather().getIndices()).astype(INDEX_TYPE)

        # note: index set sort method only sorts locally on each processor

        # note: output of argsort is what we want for MatPermute(); it
        # maps from indices in the sorted array, to indices in the original
        # unsorted array.  

        # use slices [Istart:Iend] of the result from argsort to create
        # a new IS that can be used as a petsc ordering
        retval = PETSc.IS(self.comm)
        retval.createGeneral(bigIndices[Istart:Iend],comm=self.comm)

        return retval

# abstract class representing a scalar basis of functions on a manifold for
# which we assume that each point has unique coordinates.  
class AbstractScalarBasis(abc.ABC):

    """
    Abstraction defining the behavior of a collection of scalar basis 
    functions, defined on a manifold for which each point has unique 
    coordinates.
    """

    @abc.abstractmethod
    def getNodesAndEvals(self, xi):
        """
        Given a parametric point ``xi``, return a list of the form
        
        ``[[index0, N_index0(xi)], [index1,N_index1(xi)], ... ]``
        
        where ``N_i`` is the ``i``-th basis function.
        """
        pass

    @abc.abstractmethod
    def get_ncp(self) -> int:
        """
        Returns the total number of basis functions.
        """
        pass

    @abc.abstractmethod
    def generate_mesh(self, comm=dolfin.MPI.comm_world) -> dolfin.Mesh:
        """
        Parameters
        ----------
        comm: MPI communicator used in finite element mesh creation.

        Returns
        -------
        Finite element mesh sufficient for extracting this spline basis
        """
        pass

    @abc.abstractmethod
    def get_degree(self) -> int:
        """
        Returns
        -------
        The polynomial degree for FEs that is sufficient for extracting this
        spline basis.
        """
        pass

    # TODO: get rid of the DG stuff in coordinate chart splines, since
    # getNodesAndEvals() is inherently unstable for discontinuous functions
    # and some other instantiation of AbstractExtractionGenerator
    # is needed to reliably handle $C^{-1}$ splines.

    #@abc.abstractmethod
    # assume DG unless this is overridden by a subclass (as DG will work even
    # if CG is okay (once they fix DG for quads/hexes at least...))
    def needs_dg(self) -> bool:
        """
        Returns
        -------
        Indicate whether or not DG elements are needed to represent this
        spline space (i.e., whether or not the basis is discontinuous).
        """
        return True

    @abc.abstractmethod
    def use_rectangular_elements(self):
        """
        Returns a Boolean indicating whether or not rectangular (i.e., quad
        or hex) elements should be used for extraction of this basis.
        """
        pass

    #@abc.abstractmethod
    #def getParametricDimension(self):
    #    return

    # Override this in subclasses to optimize memory use.  It should return
    # the maximum number of IGA basis functions whose supports might contain
    # a finite element node (i.e, the maximum number of nonzero
    # entries in a row of M corrsponding to that FE basis function.)
    def get_prealloc(self):
        """
        Returns some upper bound on the number of nonzero entries per row
        of the extraction matrix for this spline space.  If this can be
        easily estimated for a specific spline type, then this method 
        should almost certainly be overriden by that subclass for memory
        efficiency, as the default value implemented in the abstract class is
        overkill.
        """
        return DEFAULT_PREALLOC

# interface needed for a control mesh with a coordinate chart
class AbstractControlMesh(abc.ABC):
    """
    Abstraction representing the behavior of a control mesh, i.e., a mapping
    from parametric to physical space.
    """

    @abc.abstractmethod
    def get_homogeneous_coordinate(self, node, direction):
        """
        Returns the ``direction``-th homogeneous component of the control 
        point with index ``node``.
        """
        pass

    @abc.abstractmethod
    def get_scalar_spline(self) -> AbstractScalarBasis:
        """
        Parameters
        ----------
        field: Field index.

        Returns
        -------
        The corresponding ``AbstractScalarBasis`` that represents the
        homogeneous component of the control mapping. If field is -1 the
        basis for the scalar space of the control mesh is returned.
        """
        #TODO separate this control mesh/scalar basis functionality
        pass

    @abc.abstractmethod
    def get_nsd(self) -> int:
        """
        Returns
        -------
        Dimension of physical space.
        """
        pass


class AbstractMultiFieldSpline(AbstractCoordinateChartSpline):
    """
    Interface for a general multi-field spline.  The reason this is a special
    case of ``AbstractCoordinateChartSpline`` (instead of being redundant in
    light of AbstractExtractionGenerator) is that it uses a collection of
    ``AbstractScalarBasis`` objects, whose ``getNodesAndEvals()`` methods
    require parametric coordinates to correspond to unique points.
    """

    @abc.abstractmethod
    def get_control_mesh(self) -> AbstractControlMesh:
        """
        Returns
        -------
        This spline's ``AbstractControlMesh`` implementation.
        """
        pass

    @abc.abstractmethod
    def get_field_spline(self, field: int) -> AbstractScalarBasis:
        """
        Parameters
        ----------
        field: Field index

        Returns
        -------
        The field index's ``AbstractScalarBasis``.
        """
        pass

    def get_prealloc(self, control):
        if control:
            retval = self.get_scalar_spline(-1).get_prealloc()
        else:
            max_prealloc = 0
            for i in range(0, self.get_num_fields()):
                prealloc = self.get_scalar_spline(i).get_prealloc()
                if prealloc > max_prealloc:
                    max_prealloc = prealloc
            retval = max_prealloc
        return retval

    def get_scalar_spline(self, field: int):
        if field == -1:
            return self.get_control_mesh().get_scalar_spline()
        else:
            return self.get_field_spline(field)

    def get_nsd(self) -> int:
        """
        Returns
        -------
        Dimension of physical space.
        """
        return self.get_control_mesh().get_nsd()

    def get_homogeneous_coordinate(self, node, direction):
        """
        Invokes the synonymous method of its control mesh.
        """
        return self.get_control_mesh() \
            .get_homogeneous_coordinate(node, direction)

    def getNodesAndEvals(self, x, field):
        return self.get_scalar_spline(field).getNodesAndEvals(x)

    def generate_mesh(self):
        return self.get_scalar_spline(-1).generate_mesh(comm=self.comm)

    def get_degree(self, field: int) -> int:
        """
        Parameters
        ----------
        field: Field index

        Returns
        -------
        Polynomial degree needed to extract the unknown scalar field
        """
        return self.get_scalar_spline(field).get_degree()

    def get_ncp(self, field: int):
        """
        Parameters
        ----------
        field: Field index

        Returns
        -------
        Number of degrees of freedom for the given ``field``.
        """
        return self.get_scalar_spline(field).get_ncp()

    def use_dg(self):
        for i in range(-1, self.get_num_fields()):
            if self.get_scalar_spline(i).needs_dg():
                return True
        return False


class EqualOrderSpline(AbstractMultiFieldSpline):
    """
    A concrete subclass of ``AbstractMultiFieldSpline`` to cover the common
    case of multi-field splines in which all unknown scalar fields are 
    discretized using the same ``AbstractScalarBasis``.

    Note
    ----
    This is the common case of all control functions and fields belonging to the
    same scalar space. Fields are all stored in homogeneous format, i.e.,
    they need to be divided through by weight to get an iso-parametric
    formulation.
    """

    def __init__(self,
                 num_fields: int,
                 control_mesh: AbstractControlMesh,
                 comm=dolfin.MPI.comm_world):
        """
        Parameters
        ----------
        num_fields: number of unknown scalar fields
        control_mesh: mapping from parametric to physical space
        comm: MPI communicator

        Note
        ----
        The control mesh provides the scalar basis to be used for all unknown
        scalar fields
        """
        self.num_fields = num_fields
        self.control_mesh = control_mesh
        super().__init__(comm)

    def get_num_fields(self):
        return self.num_fields

    def get_control_mesh(self):
        return self.control_mesh

    def get_field_spline(self, field):
        return self.get_scalar_spline(-1)

    def add_zero_dofs_by_location(self, subdomain: dolfin.SubDomain,
                                  field: int):
        """
        In the equal-order case there is a one-to-one correspondence between
        the DoFs of the scalar fields and the control points of the
        geometrical mapping.

        This method assigns homogeneous Dirichlet BCs to DoFs of the given
        ``field`` if the corresponding control points fall within the
        geometric definition of the ``subdomain``.

        Parameters
        ----------
        subdomain: Geometric DOLFIN subdomain
        field: Field index
        """
        # this is prior to the permutation
        Istart, Iend = self.M_control.mat().getOwnershipRangeColumn()
        nsd = self.get_nsd()
        # since this checks every single control point, it needs to
        # be scalable
        for I in numpy.arange(Istart, Iend):
            p = numpy.zeros(nsd+1)
            for j in numpy.arange(0,nsd+1):
                p[j] = self.get_homogeneous_coordinate(I, j)
            for j in numpy.arange(0,nsd):
                p[j] /= p[nsd]
            # make it strictly based on location, regardless of how the
            # on_boundary argument is handled
            isInside = subdomain.inside(p[0:nsd],False) \
                       or subdomain.inside(p[0:nsd],True)
            if isInside:
                self.zero_dofs += [self.globalDof(field, I), ]


class FieldListSpline(AbstractMultiFieldSpline):
    """
    A concrete case of a multi-field spline that is constructed from a given
    list of distinct ``AbstractScalarBasis`` objects.
    """

    def __init__(self, control_mesh: AbstractControlMesh,
                 fields: typing.List[AbstractScalarBasis]):
        """
        Parameters
        ----------
        control_mesh: Mapping from parametric to physical space
        fields: List of scalar bases for the unknown scalar fields
        """
        self.control_mesh = control_mesh
        self.fields = fields
        super().__init__(dolfin.MPI.comm_world)

    def get_num_fields(self):
        return len(self.fields)

    def get_control_mesh(self):
        return self.control_mesh

    def get_field_spline(self, field):
        return self.fields[field]
