from __future__ import division
import numpy as np
import mapOnInterval as moi
import mapOnInterval2d as moi2d
import time
from measures import  *
from math import pi
import matplotlib.pyplot as plt
import scipy
from fenics import *

tol = 1E-14

class linEllipt():
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

class linEllipt2d():
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
		return moi2d.mapOnInterval("expl", vals)
		

if __name__ == "__main__":
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
		u_D = Expression('(x[0] >= 0.5 && x[1] <= 0.6) ? 1 : 0', degree=2)
		
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

		class fTestbed(Expression): # more complicated source and sink term
			def eval(self, values, x):
				if pow(x[0]-0.6, 2) + pow(x[1]-0.85, 2) <= 0.1*0.1:
					values[0] = -20
				elif pow(x[0]-0.2, 2) + pow(x[1]-0.75, 2) <= 0.1*0.1:
					values[0] = 20
				else:
					values[0] = 0
		
		def boundary(x, on_boundary):
			return on_boundary
			
		def boundaryD(x, on_boundary): # special Dirichlet boundary condition
			if on_boundary:
				if x[0] >= 0.6-tol and x[1] <= 0.5:
					return True
				elif x[0] <= tol: # obsolete
					return True
				else:
					return False
			else:
				return False
		f = fTestbed(degree = 2)
		lE2d = linEllipt2d(f, u_D, boundaryD)
		k = myKappaTestbed(degree = 2)
		uSol = lE2d.solve(k)
		
		plt.contourf(uSol.values, 30)
		plt.show()


		
		
		
		
		
		
