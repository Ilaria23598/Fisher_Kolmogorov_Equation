import mshr
import meshio
import math
import dolfin
from mpi4py import MPI
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import pyrameters as PRM
import os
import sys
import getopt
import pandas as pd

cwd = os.getcwd()
#sys.path.append(cwd)
sys.path.append(cwd + '/../utilities')


import ParameterFile_handler as prmh
import BCs_handler
import Mesh_handler
import XDMF_handler
import TensorialDiffusion_handler
import Solver_handler
import HDF5_handler
import Common_main

import ErrorComputation_handler


# PROBLEM CONVERGENCE ITERATIONS
def problemconvergence(filename, conv):

	errors = pd.DataFrame(columns = ['Error_L2_u','Error_DG_u'])

	for it in range(0,conv):
		# Convergence iteration solver
		errors = problemsolver(filename, it, True, errors)
		errors.to_csv("solution/ErrorsDGP4P5.csv")


# PROBLEM SOLVER
def DirichletBoundary(X, param, BoundaryID, time, mesh):

	# Vector initialization
	bc = []

	# Skull Dirichlet BCs Imposition
	period = param['Temporal Discretization']['Problem Periodicity']
		
	if param['Boundary Conditions']['Skull BCs'] == "Dirichlet" :

		# Boundary Condition Extraction Value
		BCsType = param['Boundary Conditions']['Input for Skull BCs']
		BCsValueX = param['Boundary Conditions']['Skull Dirichlet BCs Value (Displacement x-component)']
		
		BCsColumnNameX = param['Boundary Conditions']['File Column Name Skull BCs (x-component)']
		

		# Control of problem dimensionality
		if (mesh.ufl_cell() == triangle):
			BCs = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)

		else:
			BCs = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)

		# Boundary Condition Imposition
		bc.append(DirichletBC(X, BCs, BoundaryID, 1))
		
		return bc

def problemsolver(filename, iteration = 0, conv = False, errors = False):

	# Import the parameters given the filename
	param = prmh.readprmfile(filename)

	parameters["ghost_mode"] = "shared_facet"

	# Handling of the mesh
	mesh = Mesh_handler.MeshHandler(param, iteration)

	# Importing the Boundary Identifier
	BoundaryID = BCs_handler.ImportFaceFunction(param, mesh)

	# Computing the mesh dimensionality
	D = mesh.topology().dim()

	# Concentration Functional Spaces
	if param['Spatial Discretization']['Method'] == 'DG-FEM':
		P1 = FiniteElement('DG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree']))

	elif param['Spatial Discretization']['Method'] == 'CG-FEM':
		P1 = FiniteElement('CG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree']))


	# Mixed FEM Spaces
	element = P1

	# Connecting the FEM element to the mesh discretization
	X = FunctionSpace(mesh, element)

	# Construction of tensorial space
	X9 = TensorFunctionSpace(mesh, "DG", 0)

	# Diffusion tensors definition
	if param['Model Parameters']['Isotropic Diffusion'] == 'No':

		K = Function(X9)
		K = TensorialDiffusion_handler.ImportPermeabilityTensor(param, mesh, K)

	else:
		K = False

	# Time step and normal definitions
	dt = param['Temporal Discretization']['Time Step']
	T = param['Temporal Discretization']['Final Time']

	n = FacetNormal(mesh)
	if param['Model Parameters']['c or l']=='c':
	
	# Solution functions definition
	        x = Function(X)
	        c = TrialFunction(X)

	# Test functions definition
	        v = TestFunction(X)

	# Previous timestep functions definition
	        c_n = Function(X)
	
	# Measure with boundary distinction definition
	
	        ds_vent, ds_skull = BCs_handler.MeasuresDefinition(param, mesh, BoundaryID)

	# Time Initialization
	        t = 0.0

	# Initial Condition Construction
	
	        x = InitialConditionConstructor(param, mesh, X, x, c_n)
	
	# Output file name definition
	        if conv:

		        OutputFN = param['Output']['Output XDMF File Name'] + '_Ref' + str(iteration) + '_'
	        else:

		        OutputFN = param['Output']['Output XDMF File Name']

	# Save the time initial solution
	        XDMF_handler.FKSolutionSave(OutputFN,int(param['Spatial Discretization']['Polynomial Degree']), x, t, param['Temporal Discretization']['Time Step'],param['Model Parameters']['c or l'],mesh)

	# Time advancement of the solution
	        c_n.assign(x)

	# Problem Resolution Cicle
	        while t < T:

		# Temporal advancement
		        t += param['Temporal Discretization']['Time Step']

		# Dirichlet Boundary Conditions vector construction
		# bc = DirichletBoundary(X, param, BoundaryID, t, mesh)

		
		        if param['Model Parameters']['N or D']=='D':
		        
		                a, L = VariationalFormulationuD(param, c, v, dt, n, c_n, K, t, mesh, ds_vent, ds_skull)
		        else:
		                a, L = VariationalFormulationuN(param, c, v, dt, n, c_n, K, t, mesh, ds_vent, ds_skull)
		
		# Linear System Resolution
		
		        x = Solver_handler.LinearSolver(a, x, L, param)
			
		# Save the solution at time t
		        XDMF_handler.FKSolutionSave(OutputFN,int(param['Spatial Discretization']['Polynomial Degree']), x, t, param['Temporal Discretization']['Time Step'],param['Model Parameters']['c or l'],mesh)
		        #values=plot(x)
		        #plt.colorbar(values)
		        #plt.show()

		        if (MPI.comm_world.Get_rank() == 0):
			         print("Problem at time {:.6f}".format(t), "solved")
			
		# Time advancement of the solution
		        c_n.assign(x)
	     
	        #if param['Domain Definition']['Type of Mesh']=='Built-in':
		       # values=plot(x)
		       # plt.colorbar(values)
		       # plt.show()
	# Error of approximation
	        if conv:
		        errors = ErrorComputation_handler.FisherKolm_Errors(param, x, errors, mesh, iteration, t, n)
		
		        values=plot(x)
		        plt.colorbar(values)
		        plt.show()

		        if (MPI.comm_world.Get_rank() == 0):
			         print(errors)

		        return errors
		        
		        
	else:
                 x = Function(X)
                 l = TrialFunction(X)
                 
                 # Test functions definition
                 v = TestFunction(X)
                 
                 # Previous timestep functions definition
                 l_n = Function(X)
                 
                 # Measure with boundary distinction definition
                 ds_vent, ds_skull = BCs_handler.MeasuresDefinition(param, mesh, BoundaryID)
                 
                 # Time Initialization
                 t = 0.0
                 
                  # Initial Condition Construction
                 l = InitialConditionConstructor(param, mesh, X, l, l_n)
                 
                 # Output file name definition
                 if conv:
                         OutputFN = param['Output']['Output XDMF File Name'] + '_Ref' + str(iteration) + '_'
                 else:   OutputFN = param['Output']['Output XDMF File Name']
                 
                 # Save the time initial solution
                 XDMF_handler.FKSolutionSave(OutputFN,int(param['Spatial Discretization']['Polynomial Degree']), x, t, param['Temporal Discretization']['Time Step'],param['Model Parameters']['c or l'],mesh)
                 
                 # Time advancement of the solution
                 l_n.assign(l)
                  # Problem Resolution Cicle
                 while t < T:
                 
                  # Temporal advancement
                  t += param['Temporal Discretization']['Time Step']
                  
                  # Variational Formulation Construction
                  a, L = VariationalFormulationlN(param, l, v, dt, n, l_n, K, t, mesh, ds_vent, ds_skull)
                  #Non linear system resolution
                  x = Solver_handler.NonLinearSolver(a, x, l, L, param)
                  if (np.mod(t,0.1)==0):
                  	XDMF_handler.FKSolutionSave(OutputFN, int(param['Spatial Discretization']['Polynomial Degree']),x, t, param['Temporal Discretization']['Time Step'],param['Model Parameters']['c or l'],mesh)
                  if (MPI.comm_world.Get_rank() == 0):
                          print("Problem at time {:.6f}".format(t), "solved")
                  # Time advancement of the solution
                          
                  l_n.assign(x)
                  
                  # Error of approximation
                  if conv:
                          errors = ErrorComputation_handler.FisherKolm_Errors(param, x, errors, mesh, iteration, t, n)
                          
                          
                          value=plot(exp(x))
                          plt.colorbar(value)
                          plt.show()
                          if (MPI.comm_world.Get_rank() == 0):
                                  print(errors)
                          return errors
#########################################################################################################################
#						Variational Formulation Definition					#
#########################################################################################################################

def VariationalFormulationlN(param, l, v, dt, n, l_n, K, time, mesh, ds_vent, ds_skull):


	time_prev = time-param['Temporal Discretization']['Time Step']
	theta = param['Temporal Discretization']['Theta-Method Parameter']
	reg = param['Spatial Discretization']['Polynomial Degree']

	h = CellDiameter(mesh)


	# Growing Coefficient Extraction
	alpha = Constant(param['Model Parameters']['alpha'])
	eps=1e-10
	

	# Diffusion Parameter Extraction
	d = Constant(param['Model Parameters']['Diffusion'])
	
	f=Expression((param['Model Parameters']['Forcing Terms']),degree=6, alpha=alpha, d=d,t=time)
	
	#f_n=Expression((param['Model Parameters']['Forcing Terms']),degree=6, t=time_prev)

	# EQUATION  CG
	
	a= exp(l)*v*dx + d*dt*dot(exp(l)*grad(l), grad(v))*dx+eps*l*v*dx - alpha*dt*exp(l)*(1-exp(l))*v*dx 
	
	L= (exp(l_n)*v + dt*f*v)*dx 
	
	# DISCONTINUOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		
		gamma=Constant(param['Model Parameters']['gamma'])
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	# EQUATION DG
		
		a= a - d*dt*dot(avg(exp(l)*grad(l)), jump(v, n))*dS - dt*d*dot(jump(l, n), avg(exp(l)*grad(v)))*dS + dt*d*gamma*reg*reg/h_avg*dot(jump(l, n), jump(v, n))*dS \
		

		
	return a, L


def VariationalFormulationuD(param, c, v, dt, n, c_n, K, time, mesh, ds_vent, ds_skull):


	time_prev = time-param['Temporal Discretization']['Time Step']
	theta = Constant(param['Temporal Discretization']['Theta-Method Parameter'])
	reg = param['Spatial Discretization']['Polynomial Degree']

	h = CellDiameter(mesh)
	h_avg = 2*(h('+')*h('-'))/(h('+')+h('-'))

	# Growing Coefficient Extraction
	alpha = Constant(param['Model Parameters']['alpha'])
	

	# Diffusion Parameter Extraction
	d = Constant(param['Model Parameters']['Diffusion'])
	f=Expression((param['Model Parameters']['Forcing Terms']),degree=6,d=d, alpha=alpha, t=time)
	f_n=Expression((param['Model Parameters']['Forcing Terms']),degree=6, d=d,alpha=alpha,t=time_prev)

	

	# EQUATION  CG    
      
	a=c*v*dx+d*theta*dt*dot(grad(c),grad(v))*dx-alpha*theta*dt*c*(1-c_n)*v*dx
	
	
	L= (c_n+ dt*theta*f + dt*(1-theta)*f_n)*v*dx - d*(1-theta)*dt*dot(grad(c_n), grad(v))*dx + alpha*(1-theta)*dt*c_n*(1-c_n)*v*dx
	
	

	# DISCONTINUOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		
		gamma=Constant(param['Model Parameters']['gamma'])

		# EQUATION DG
      
		a= a - dt*theta*dot(avg(grad(v)), jump(c, n))*dS - dt*theta*dot(jump(v, n), avg(grad(c)))*dS \
			+ dt*theta*gamma*reg*reg/h_avg*dot(jump(v, n), jump(c, n))*dS- dt*theta*dot(grad(v), n)*c*ds- dt*theta*dot(n, grad(c))*v*ds  \
      + dt*theta*(gamma*reg*reg/h)*v*c*ds
      
		L= L + (dt*(1-theta)*dot(avg(grad(v)), jump(c_n, n))*dS) + dt*(1-theta)*dot(jump(v, n), avg(grad(c_n)))*dS - dt*(1-theta)*gamma*reg*reg/h_avg*dot(jump(v, n), jump(c_n, n))*dS + dt*(1-theta)*dot(grad(v), n)*c_n*ds + dt*(1-theta)*dot(n, grad(c_n))*v*ds - dt*(1-theta)*(gamma*reg*reg/h)*v*c_n*ds
	        


	return a, L
	
def VariationalFormulationuN(param, c, v, dt, n, c_n, K, time, mesh, ds_vent, ds_skull):


	time_prev = time-param['Temporal Discretization']['Time Step']
	theta = Constant(param['Temporal Discretization']['Theta-Method Parameter'])
	reg = param['Spatial Discretization']['Polynomial Degree']

	h = CellDiameter(mesh)
	h_avg = 2*(h('+')*h('-'))/(h('+')+h('-'))

	# Growing Coefficient Extraction
	alpha = Constant(param['Model Parameters']['alpha'])
	
	# Diffusion Parameter Extraction
	d = Constant(param['Model Parameters']['Diffusion'])
	
	# Forcing terms Extraction
	f=Expression((param['Model Parameters']['Forcing Terms']),degree=6,d=d, alpha=alpha, t=time)
	f_n=Expression((param['Model Parameters']['Forcing Terms']),degree=6, d=d,alpha=alpha,t=time_prev)

	
	# EQUATION  CG    
      
	a=c*v*dx+d*theta*dt*dot(grad(c),grad(v))*dx-alpha*theta*dt*c*(1-c_n)*v*dx
	
	
	L= (c_n+ dt*theta*f + dt*(1-theta)*f_n)*v*dx - d*(1-theta)*dt*dot(grad(c_n), grad(v))*dx + alpha*(1-theta)*dt*c_n*(1-c_n)*v*dx
	
	

	# DISCONTINUOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		
		gamma=Constant(param['Model Parameters']['gamma'])

		# EQUATION DG
      
		a= a - dt*d*theta*dot(avg(grad(v)), jump(c, n))*dS - dt*d*theta*dot(jump(v, n), avg(grad(c)))*dS \
			+ dt*d*theta*gamma*reg*reg/h_avg*dot(jump(v, n), jump(c, n))*dS
      
		L= L + (dt*d*(1-theta)*dot(avg(grad(v)), jump(c_n, n))*dS) + dt*(1-theta)*d*dot(jump(v, n), avg(grad(c_n)))*dS 
	        


	return a, L	

##############################################################################################################################
#				Constructor of Initial Condition from file or constant values 				     #
##############################################################################################################################

def InitialConditionConstructor(param, mesh, X, c, c_n):

	# Solution Initialization
	c0 = param['Model Parameters']['Initial Condition']

	if (mesh.ufl_cell()==triangle):
		x0 = Expression((c0),degree=5)

	else:
		x0 = Expression((c0),degree=5)

	c = interpolate(x0, X)

	# Initial Condition Importing from Files


	if param['Model Parameters']['Initial Condition from File'] == 'Yes':

		c_n = HDF5_handler.ImportICfromFile(param['Model Parameters'] ['Initial Condition File Name'], mesh, c_n, param['Model Parameters']['Name of IC Function in File'])
		ccn=np.ones(np.size(c_n.vector()[:]))
		if param['Model Parameters']['c or l']=='l':
		     c_n.vector()[:] = -0.7*ccn+c_n.vector()[:]
		else: 
		     c_n.vector()[:] = 0.2*c_n.vector()[:]
		
		assign(c, c_n)

	return c


######################################################################
#				Main 				     #
######################################################################

if __name__ == "__main__":

	Common_main.main(sys.argv[1:], cwd, '/../physics/FisherKolm')

	if (MPI.comm_world.Get_rank() == 0):
		print("Problem Solved!")

