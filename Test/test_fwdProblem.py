from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from fwdProblem import *
from measures import *
from haarWavelet2d import *


RESOL_global = 6

rect = Rectangle((0,0), (180,78), resol=RESOL_global)
rect2 = Rectangle((0,0), (1,1), resol=RESOL_global)

f = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-40)**2 + (y-20)**2) < 4)*(-8.2667/40))
f2 = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-0.3)**2 + (y-0.55)**2) < 0.01)*(-100.0))


def u_D_term(x, y):
	return np.logical_and(x >= 80-tol, y <= 40)*2.0

def u_D_term2(x, y):
	return np.logical_and(x >= 0.5, y <= 0.6)*2.0

u_D2 = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term2(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] >= 90-tol and x[1] <= 30:
			return True
		elif x[0] <= 10**-8 and x[1] >= 10 and x[1] <= 40:
			return True
		else:
			return False


def boundary_D_boolean2(x): # special Dirichlet boundary condition
		if x[0] >= 0.6-tol and x[1] <= 0.5:
			return True
		elif x[0] <= 10**-8:
			return True
		else:
			return False


def boundary_D_boolean2_version(x):
	if x[1] > tol:
		return True
	else:
		return False

def u_D_term2_version(x, y):
	return 0*x

#u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))
u_D = mor.mapOnRectangle(rect, "handle", lambda x, y: np.logical_and(x < 10**-8, True)*1.0 - 2.0*np.logical_and(x > 90-10**-8, True)*1.0)

u_D2= mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term2(x,y))



"""def boundary_D_boolean(x):
	if (x[1] > 30-10**(-8) and x[1] < 50+10**(-8)) and (x[0] < 10**-8 or x[0] > 180-10**-8):
		return True
	else:
		return False"""

# 1 on left border, -2 on lower right corner and parts of adjacent borders


fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
fwd2 = linEllipt2dRectangle(rect2, f2, u_D2, boundary_D_boolean2)

k = mor.mapOnRectangle(rect, "handle", lambda x,y: 1/(180*180)*x*(180-x)+3/40*y*np.logical_and(y <= 40, True))
k2 = mor.mapOnRectangle(rect2, "handle", lambda x,y: 0.001*x*(1-x)+0.5*y*np.logical_and(y <= 0.5, True) + 2*np.exp(-40*(y-np.exp(-2.3*x))**2)) 

start = time.time()
F_, F = fwd.solve(k, pureFenicsOutput="both")
end = time.time()
print("solve1:")
print(end-start)

start = time.time()
F2_, F2 = fwd2.solve(k2, pureFenicsOutput="both")
end = time.time()
print("solve2:")
print(end-start)

vtkfile = File('solution.pvd')
vtkfile << F2_
fig1 = plt.figure(); plt.ion()
ax = fig1.add_subplot(2, 1, 1, aspect="equal")
plt.contourf(F2.X, F2.Y, F2.values, 60); plt.colorbar()
ax2 = fig1.add_subplot(2, 1, 2, aspect="equal")
plt.contourf(k2.x, k2.y, k2.values, 60);plt.colorbar()
plt.show()

fig2 = plt.figure(); plt.ion()
ax = fig2.add_subplot(2, 1, 1, aspect="equal")
plt.contourf(F.X, F.Y, F.values, 60); plt.colorbar()
ax2 = fig2.add_subplot(2, 1, 2, aspect="equal")
plt.contourf(k.x, k.y, k.values, 60);plt.colorbar()
plt.show()

kappa = morToFenicsConverterHigherOrder(k, fwd2.mesh, fwd2.V)
vtkfile = File('kappa.pvd')
vtkfile << kappa
"""V = fwd.V
mesh = fwd.mesh
u = TrialFunction(V)
v = TestFunction(V)
L = self.f*v*dx		
		a = k*dot(grad(u), grad(v))*dx
		uSol = Function(self.V)
		solve(a == L, uSol, self.bc)"""
"""m1 = GeneralizedGaussianWavelet2d(rect, 1.0, 1.5, 5)
u1 = m1.sample()
plt.figure();
plt.contourf(u1.values)
plt.colorbar()
k1 = mor.mapOnRectangle(rect, "handle", lambda x,y: np.exp(u1.handle(x, y)))
F1_ = fwd.solve(k1, pureFenicsOutput=True)
F1 = fwd.solve(k1)
vtkfile = File('solution_sandbox.pvd')
vtkfile << F1_

plt.figure(); plt.ion()
plt.contourf(F1.X, F1.Y, F1.values, 60); plt.colorbar()
plt.show()"""



