from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from fwdProblem import *
from measures import *
from haarWavelet2d import *

rect = Rectangle((0,0), (1,1), resol=6)

def myUTruth(x, y):
		"""if x[0] <= 0.5 +tol  and x[0] >= 0.45 - tol and x[1] <= 0.5+tol:
			return -4
		elif x[0] <= 0.5+tol and x[0] >= 0.45 - tol and x[1] >= 0.6 - tol:
			return -4
		elif x[0] <= 0.75 + tol and x[0] >= 0.7 - tol and x[1] >= 0.2 - tol and x[1] <= 0.8+tol:
			return 2
		else:
			return 0"""
		#if x.ndim == 1 and x.shape[0] == 2:
		return 2*np.logical_and(y <= 0.4-0.1*x, True) + (-1)*np.logical_and(y>0.4-0.1*x, y<=0.8*x) + (-3)*np.logical_and(y > 0.4-0.1*x, np.logical_and(y > 0.8*x, y<=0.7-0.15*x))
def u_D_term(x, y):
	return x*0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[1] <= tol:
			return True
		else:
			return False
#f = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)# (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)
f1 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-0.25)**2 + (y-0.5)**2 < 0.05**2)*(-20.0))
f2 = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-0.25)**2 + (y-0.25)**2) < 0.05**2)*(-20.0))




fwd = linEllipt2dRectangle_hydrTom(rect, [f1,f2], u_D, boundary_D_boolean)

k = mor.mapOnRectangle(rect, "handle", lambda x,y: np.exp(myUTruth(x,y)))

F_, F = fwd.solve(k, pureFenicsOutput="both")
vtkfile = File('solution0.pvd')
vtkfile << F_[0]
vtkfile = File('solution1.pvd')
vtkfile << F_[1]
plt.figure(); plt.ion()
plt.contourf(F[0].X, F[0].Y, F[0].values, 60); plt.colorbar()
plt.show()

plt.figure(); plt.ion()
plt.contourf(F[1].X, F[1].Y, F[1].values, 60); plt.colorbar()
plt.show()

plt.figure(); plt.ion()
plt.contourf(k.X, k.Y, np.log(k.values), 60); plt.colorbar()
plt.show()



