from __future__ import division
import numpy as np
import mapOnInterval as moi
#import mapOnInterval2d as moi2d
import time
from measures import  *
from math import pi
from rectangle import *
import matplotlib.pyplot as plt
import scipy
from fenics import *

tol = 1E-14

class linEllipt():
	# 1d class for the linear elliptical forward problem; might get replaced by fenics routine soon
	# model: -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
	def __init__(self, g, pplus, pminus):
		self.g = g
		self.pplus = pplus
		self.pminus = pminus

	def solve(self, x, k, g = None, pplus=None, pminus=None, returnC = False, moiMode = False):
	# solves -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
	# if g == None, we take self.g, else we take this as the right hand side (same for pplus and pminus)
		if moiMode == True: #do not use this!
			
			kinv = moi.mapOnInterval("handle", lambda x: 1/k.handle(x))
			I_1 = moi.integrate(x, kinv)
			if g == None:
				I_2 = moi.integrate(x, self.g)
			else:
				I_2 = moi.integrate(x, g)	
			if pplus == None:
				pplus = self.pplus
			if pminus == None:
				pminus = self.pminus
			I_2timeskinv = moi.mapOnInterval("expl", I_2.values/k.values)
			I_3 = moi.integrate(x, I_2timeskinv)
			C = (pplus - pminus + I_3.values[-1])/(I_1.values[-1])
			p = moi.mapOnInterval("expl", -I_3.values + C*I_1.values + pminus)
		else: # non-moi mode:
			if isinstance(k, moi.mapOnInterval):
				kvals = k.values
			else:
				kvals = k
			if g == None:
				g = self.g
			if pplus == None:
				pplus = self.pplus
			if pminus == None:
				pminus = self.pminus
			gvals = g.values
			I_1 = np.concatenate((np.array([0]), scipy.integrate.cumtrapz(1/kvals, x)))
			I_2 = np.concatenate((np.array([0]), scipy.integrate.cumtrapz(gvals, x)))
			I_3 = np.concatenate((np.array([0]), scipy.integrate.cumtrapz(I_2/kvals, x)))
			C = (pplus - pminus + I_3[-1])/(I_1[-1])
			p = moi.mapOnInterval("expl", -I_3 + C*I_1 + pminus)
		
		if returnC:
			return p, C
		else:
			return p

def morToFenicsConverter(f, mesh, V):
	# converts a mapOnRectangle function to a fenics function on a mesh. Needed for compatibility
	coords = mesh.coordinates().T

	# evaluate function in vertices
	vals = f.handle(coords[0, :], coords[1, :])
	if not vals.dtype == np.float_:
		raise Exception("make sure your function returns float values (i.e. no integers or anything else)")

	fnc = Function(V)
	fnc.vector().set_local(vals[dof_to_vertex_map(V)])
	return fnc

def morToFenicsConverterHigherOrder(f, mesh, V):
	# get coordinates of all degrees of freedom
	dof_coord_raw = V.tabulate_dof_coordinates()
	dof_coord_x = np.array([dof_coord_raw[2*k] for k in range(len(dof_coord_raw)//2)]).reshape((-1,1))
	dof_coord_y = np.array([dof_coord_raw[2*k+1] for k in range(len(dof_coord_raw)//2)]).reshape((-1,1))
	dof_coords = np.concatenate((dof_coord_x, dof_coord_y), axis=1)
	
	# compute function values on all degrees of freedom
	vals = f.handle(dof_coords[:, 0], dof_coords[:, 1])
	
	# kind of like dof_to_vertex_map
	dofs = V.dofmap().dofs()
	fnc = Function(V)
	fnc.vector().set_local(vals[dofs])
	return fnc
	
class linEllipt2dRectangle():
	# main class for the linear elliptical 2d problem on a rectangular domain
	# can handle several kinds of PDE operations which are needed by higher-level classes
	# in principle this should be the only class in the inverse problem setting that can "see" fenics functionality
	# (unless for debugging or testing purposes)
	def __init__(self, rect, f, u_D, boundary_D_boolean):
		assert isinstance(rect, Rectangle)
		self.rect = rect
		self.mesh = RectangleMesh(Point(rect.x1,rect.y1), Point(rect.x2,rect.y2), 2**rect.resol, 2**rect.resol)
		self.V = FunctionSpace(self.mesh, 'P', 5)
		
		# if the forcing term and/or the dirichlet boundary data are not already in fenics type, convert
				
		if isinstance(f, mor.mapOnRectangle):
			self.f = morToFenicsConverterHigherOrder(f, self.mesh, self.V)
		else:
			self.f = f
		
		if isinstance(u_D, mor.mapOnRectangle):
			self.u_D = morToFenicsConverterHigherOrder(u_D, self.mesh, self.V)
		else:
			self.u_D = u_D
		
		self.boundary_markers = FacetFunction("size_t", self.mesh)

		# the following implements Dirichlet boundary conditions with value u_D on the boundary specified by boundary_D_boolean 
		# (and the pre-implemented on_boundary functionality)
		# the rest is assumed 0-Neumann
		
		class BoundaryDirichlet(SubDomain):
			tol = 1E-14
			def inside(self, x, on_boundary):
				return on_boundary and boundary_D_boolean(x)
		class BoundaryNeumann(SubDomain):
			tol = 1E-14
			def inside(self, x, on_boundary):
				return on_boundary and not boundary_D_boolean(x)


		bD = BoundaryDirichlet()
		bD.mark(self.boundary_markers, 1)
		bN = BoundaryNeumann()
		bN.mark(self.boundary_markers, 2)

		boundary_conditions = {1: {'Dirichlet': self.u_D}, 2: {'Neumann':   Constant(0.0)}}

		bcs = []
		for i in boundary_conditions:
			if 'Dirichlet' in boundary_conditions[i]:
				bc = DirichletBC(self.V, boundary_conditions[i]['Dirichlet'], self.boundary_markers, i)
				bcs.append(bc)
		
		self.bc = bcs				
	
	def solve(self, k, pureFenicsOutput=False):	# solves -div(k*nabla(y)) = f for y	with b.c. as specified in initialization
		set_log_level(40)
		if isinstance(k, mor.mapOnRectangle):
			k = morToFenicsConverterHigherOrder(k, self.mesh, self.V)
		
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		L = self.f*v*dx		
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		solve(a == L, uSol, self.bc)
		if pureFenicsOutput:
			return uSol
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.rect.resol+1, 2**self.rect.resol+1))
		if pureFenicsOutput == "Both":
			return uSol, mor.mapOnRectangle(self.rect, "expl", vals[0:-1,0:-1]) #cut vals to fit in rect grid
		else:
			return mor.mapOnRectangle(self.rect, "expl", vals[0:-1,0:-1]) #cut vals to fit in rect grid  
	
	def evalInnerProdListPhi(self, phis, u, v): # computes \int phi * nabla(u)*nabla(v) for all phi in phis
		lst = []
		if isinstance(u, mor.mapOnRectangle):
			u = morToFenicsConverterHigherOrder(u, self.mesh, self.V)
		if isinstance(v, mor.mapOnRectangle):
			v = morToFenicsConverterHigherOrder(v, self.mesh, self.V)
		for phi in phis:
			if isinstance(phi, mor.mapOnRectangle):
				phi = morToFenicsConverterHigherOrder(phi, self.mesh, self.V)			
			lst.append(assemble(phi*dot(grad(u),grad(v))*dx))
		return lst
		
	
	def solveWithDiracRHS(self, k, ws, xs, pureFenicsOutput=False): # solves -div(k*nabla(y)) = sum_i w_i*dirac_{x_i} with homogenous bcs
		set_log_level(40)
		if isinstance(k, mor.mapOnRectangle):
			k = morToFenicsConverterHigherOrder(k, self.mesh, self.V)
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		delta = []
		for w, x in zip(ws, xs):
			delta.append(PointSource(self.V, Point(x[0], x[1]), w))
		
		a = k*dot(grad(u), grad(v))*dx
		L = Constant(0)*v*dx
		A, b = assemble_system(a, L, DirichletBC(self.V, Constant(0), self.boundary_markers, 1))
		for d in delta:
			d.apply(b)
		
		uSol = Function(self.V)
		solve(A, uSol.vector(), b)
		if pureFenicsOutput:
			return uSol
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.rect.resol+1, 2**self.rect.resol+1))
		return mor.mapOnRectangle(self.rect, "expl", vals[0:-1,0:-1]) #cut vals to fit in rect grid 
	
	
	
	
	
	
	
	
	def evalInnerProd(self, k, u, v): # evaluate \int k * nabla(u)*nabla(v) over Omega
		return assemble(k*dot(grad(u),grad(v))*dx)
		
	def solveWithHminus1RHS(self, k, k1, y, pureFenicsOutput=False): # solves -div(k*nabla(y1)) = div(k1*nabla(y)) for y1		
		if isinstance(k, mor.mapOnRectangle):
			k = morToFenicsConverterHigherOrder(k, self.mesh, self.V)
		if isinstance(k1, mor.mapOnRectangle):
			k1 = morToFenicsConverterHigherOrder(k1, self.mesh, self.V)
		if isinstance(y, mor.mapOnRectangle):
			y = morToFenicsConverterHigherOrder(y, self.mesh, self.V)
		set_log_level(40)
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		#L = self.f*v*dx		
		L = - k1*dot(grad(y),grad(v))*dx
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		u_D_0 = Expression('0*x[0]', degree=2)
		solve(a == L, uSol, DirichletBC(self.V, Constant(0), self.boundary_markers, 1))#DirichletBC(self.V, u_D_0, self.boundary_D))#
		
		if pureFenicsOutput:
			return uSol
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.rect.resol+1, 2**self.rect.resol+1))
		return mor.mapOnRectangle(self.rect, "expl", vals[0:-1,0:-1])
	
	def solveWithHminus1RHS_variant(self, k, k1, y1, k2, y2): # solves -div(k*nabla(y22)) = div(k1*nabla(y2) + k2*nabla(y1)) for y22	
		if isinstance(k, mor.mapOnRectangle):
			k = morToFenicsConverterHigherOrder(k, self.mesh, self.V)
		if isinstance(k1, mor.mapOnRectangle):
			k1 = morToFenicsConverterHigherOrder(k1, self.mesh, self.V)
		if isinstance(k2, mor.mapOnRectangle):
			k2 = morToFenicsConverterHigherOrder(k2, self.mesh, self.V)
		set_log_level(40)
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		#L = self.f*v*dx		
		L = - (k1*dot(grad(y2),grad(v)) + k2*dot(grad(y1),grad(v)))*dx
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		u_D_0 = Expression('0*x[0]', degree=2)
		solve(a == L, uSol, DirichletBC(self.V, Constant(0), self.boundary_markers, 1))#DirichletBC(self.V, u_D_0, self.boundary_D))
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.rect.resol+1, 2**self.rect.resol+1))
		return mor.mapOnRectangle(self.rect, "expl", vals[0:-1,0:-1])
		

"""class linEllipt2d(): # should be obsolete after linEllipt2dRectangle
	# model: -(k*p')' = f, with p = u_D on the Dirichlet boundary and Neumann = 0 on the rest 
	def __init__(self, f, u_D, boundaryD, resol=4, xresol=7):
		self.f = f
		self.u_D = u_D
		self.boundaryD = boundaryD
		self.mesh = UnitSquareMesh(2**resol, 2**resol)
		self.resol = resol
		self.V = FunctionSpace(self.mesh, 'P', 1)
		self.bc = DirichletBC(self.V, u_D, boundaryD)
		self.xresol = 7


	def solve(self, k, pureFenicsOutput=False):	# solves -div(k*nabla(y)) = f for y
		set_log_level(40)
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		L = self.f*v*dx		
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		solve(a == L, uSol, self.bc)
		if pureFenicsOutput:
			return uSol
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.resol+1, 2**self.resol+1))
		return moi2d.mapOnInterval("expl", vals)
	
	def solveWithHminus1RHS(self, k, k1, y, pureFenicsOutput=False): # solves -div(k*nabla(y1)) = div(k1*nabla(y)) for y1
		set_log_level(40)
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		#L = self.f*v*dx		
		L = - k1*dot(grad(y),grad(v))*dx
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		u_D_0 = Expression('0*x[0]', degree=2)
		solve(a == L, uSol, DirichletBC(self.V, u_D_0, self.boundaryD))#
		if pureFenicsOutput:
			return uSol
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.resol+1, 2**self.resol+1))
		return moi2d.mapOnInterval("expl", vals)
	
	def solveWithHminus1RHS_variant(self, k, k1, y1, k2, y2): # solves -div(k*nabla(y22)) = div(k1*nabla(y2) + k2*nabla(y1)) for y22
		set_log_level(40)
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		#L = self.f*v*dx		
		L = - (k1*dot(grad(y2),grad(v)) + k2*dot(grad(y1),grad(v)))*dx
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		u_D_0 = Expression('0*x[0]', degree=2)
		solve(a == L, uSol, DirichletBC(self.V, u_D_0, self.boundaryD))
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.resol+1, 2**self.resol+1))
		return moi2d.mapOnInterval("expl", vals)"""
		
"""class sandbox(): # should be obsolete after linEllipt2dRectangle or should be redefined as a special case of linEllipt2dRectangle
	# model: -(k*p')' = f, with p = u_D on the Dirichlet boundary and Neumann = 0 on the rest 
	def __init__(self, f, resol=4, xresol=7):
		self.f = f
		self.u_D = Constant('0')
		self.resol = resol
		def boundaryD(x, on_boundary): # special Dirichlet boundary condition
			if on_boundary:
				if x[1] > 10**(-8):
					return True
				else:
					return False
			else:
				return False
		
		self.boundaryD = boundaryD
		self.mesh = RectangleMesh(Point(0,0), Point(160,78), 2**self.resol, 2**self.resol)
		self.V = FunctionSpace(self.mesh, 'P', 1)
		self.bc = DirichletBC(self.V, self.u_D, self.boundaryD)
		self.xresol = 7


	def solve(self, k, pureFenicsOutput=False):	# solves -div(k*nabla(y)) = f for y
		set_log_level(40)
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		L = self.f*v*dx		
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		solve(a == L, uSol, self.bc)
		if pureFenicsOutput:
			return uSol
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.resol+1, 2**self.resol+1))
		return moi2d.mapOnInterval("expl", vals)
	
	def solveWithHminus1RHS(self, k, k1, y, pureFenicsOutput=False): # solves -div(k*nabla(y1)) = div(k1*nabla(y)) for y1
		set_log_level(40)
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		#L = self.f*v*dx		
		L = - k1*dot(grad(y),grad(v))*dx
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		u_D_0 = Expression('0*x[0]', degree=2)
		solve(a == L, uSol, DirichletBC(self.V, u_D_0, self.boundaryD))#
		if pureFenicsOutput:
			return uSol
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.resol+1, 2**self.resol+1))
		return moi2d.mapOnInterval("expl", vals)
	
	def solveWithHminus1RHS_variant(self, k, k1, y1, k2, y2): # solves -div(k*nabla(y22)) = div(k1*nabla(y2) + k2*nabla(y1)) for y22
		set_log_level(40)
		u = TrialFunction(self.V)
		v = TestFunction(self.V)
		#L = self.f*v*dx		
		L = - (k1*dot(grad(y2),grad(v)) + k2*dot(grad(y1),grad(v)))*dx
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		u_D_0 = Expression('0*x[0]', degree=2)
		solve(a == L, uSol, DirichletBC(self.V, u_D_0, self.boundaryD))
		vals = np.reshape(uSol.compute_vertex_values(), (2**self.resol+1, 2**self.resol+1))
		return moi2d.mapOnInterval("expl", vals)"""
		
"""if __name__ == "__main__":
	if False: # 1d case
		x = np.linspace(0, 1, 512)
	
		# boundary values for forward problem
		# -(k * p')' = g
		# p(0) = pminus
		# p(1) = pplus
		pplus = 2.0
		pminus = 1.0	
		# right hand side of forward problem
		g = moi.mapOnInterval("handle", lambda x: 3.0*x*(1-x))	
		# construct forward problem
		fwd = linEllipt(g, pplus, pminus)
	
		# prior measure:
		alpha = 0.7
		beta = 0.5
		mean = np.zeros((31,))
		prior = GaussianFourier(mean, alpha, beta)
	
		# case 1: random ground truth
		u0 = prior.sample()
		
		# case 2: given ground truth
		J = 9
		num = 2**J
		x = np.linspace(0, 1, 2**(J), endpoint=False)
		gg1 = lambda x: 1 + 2**(-J)/(x**2+2**J) + 2**J/(x**2 + 2**J)*np.cos(32*x)
		g1 = lambda x: gg1(2**J*x)
		gg2 = lambda x: (1 - 0.4*x**2)/(2**(J+3)) + np.sin(7*x/(2*pi))/(1 + x**2/2**J)
		g2 = lambda x: gg2(2**J*x)
		gg3 = lambda x: 3 + 3*(x**2/(2**(2*J)))*np.sin(x/(8*pi))
		g3 = lambda x: gg3(2**J*x)
		gg4 = lambda x: (x**2/3**J)*0.1*np.cos(x/(2*pi))-x**3/8**J + 0.1*np.sin(3*x/(2*pi))
		g4 = lambda x: gg4(2**J*x)
		vec1 = g2(x[0:2**(J-5/2)])
		vec2 = g1(x[2**(J-5/2):2**(J-1.5)])
		vec3 = g3(x[2**(J-1.5):2**(J)-2**(J-1.2)])
		vec4 = g4(x[2**(J)-2**(J-1.2):2**(J)])

		f = np.concatenate((vec1, vec2, vec3, vec4))
		u0 = moi.mapOnInterval("expl", f)
		k0 = moi.mapOnInterval("expl", np.exp(u0.values))
		
		lE = linEllipt(g, pplus, pminus)
		
		st = time.time()
		for m in range(100):
			p = lE.solve(x, k0, moiMode = True)
			
		et = time.time()
		print(et-st)
		st = time.time()
		for m in range(100):
			p2 = lE.solve(x, k0, moiMode = False)
		et = time.time()
		print(et-st)
		plt.show()
	else: #2d case
		#u_D = Expression('(x[0] >= 100 && x[1] <= 100) ? 1 : 0', degree=2)
		
		class myKappaTestbed(Expression): # more complicated topology
			def eval(self, values, x):
				if x[0] <= 0.5 +tol  and x[0] >= 0.45 - tol and x[1] <= 0.5+tol:
					values[0] = 0.0001
				elif x[0] <= 0.5+tol and x[0] >= 0.45 - tol and x[1] >= 0.6 - tol:
					values[0] = 0.0001
				elif x[0] <= 0.75 + tol and x[0] >= 0.7 - tol and x[1] >= 0.2 - tol and x[1] <= 0.8+tol:
					values[0] = 100
				else:
					values[0] = 1
				if x[1] <= 1.5*x[0] - 100:
					values[0] = 0.0001
				else:
					values[0] = 100
				
				#values[0] = 0.077
				if x[0] > 80:
					values[0] = 0.1
				else:
					values[0] = 0.05

		class fTestbed(Expression): # more complicated source and sink term
			def eval(self, values, x):
				if pow(x[0]-40, 2) + pow(x[1]-20, 2) <= 2**2:
					values[0] = -8.2667/100
					#elif pow(x[0]-0.2, 2) + pow(x[1]-0.75, 2) <= 0.1*0.1:
					#	values[0] = 20
				else:
					values[0] = 0
		
		def boundary(x, on_boundary):
			return on_boundary
			
		def boundaryD(x, on_boundary): # special Dirichlet boundary condition
			if on_boundary:
				if x[0] >= 1000 and x[1] <= 100:
					return True
				elif x[0] <= 0.1: # obsolete
					return True
				elif x[1] >= 499:
					return True
				else:
					return False
			else:
				return False"""
"""		f = fTestbed(degree = 2)
		lE2d = sandbox(f, resol=7)
		k = myKappaTestbed(degree = 2)
		uSol = lE2d.solve(k, pureFenicsOutput=True)
vtkfile = File('solution_sandbox.pvd')
vtkfile << uSol
		#plot(lE2d.mesh)
		def plot3d(u):
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			N2 = u.values.shape[0]
			xx = np.linspace(0, 1, N2)
			XX, YY = np.meshgrid(xx, xx)
			ax.plot_wireframe(XX, YY, u.values)
			plt.show()
		vals = np.reshape(uSol.compute_vertex_values(), (2**lE2d.resol+1, 2**lE2d.resol+1))
		fnc = moi2d.mapOnInterval("expl", vals)
		plt.figure();
		plt.ion()
		plt.contourf(fnc.values, 30)
		plt.colorbar()
		plt.show()
		plot3d(fnc)"""


		
		
		
		
		
		
