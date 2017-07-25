from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from fwdProblem import *
from measures import *
from haarWavelet2d import *


rect = Rectangle((0,0), (180,78), resol=7)

f2 = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-40)**2 + (y-20)**2) < 4)*(-8.2667/40))
"""plt.figure()
plt.subplot(2,1,1)
plt.contourf(f2.X, f2.Y, f2.values)
plt.subplot(2,1,2)
plt.contourf(f2.X, f2.Y, f2.handle(f2.X, f2.Y))	
f22 = mor.mapOnRectangle(rect, "wavelet", f2.waveletcoeffs)

plt.figure()
plt.subplot(2,1,1)
plt.contourf(f22.X, f22.Y, f22.values)
plt.subplot(2,1,2)
plt.contourf(f22.X, f22.Y, f22.handle(f22.X, f22.Y))	

f222 = mor.mapOnRectangle(rect, "handle", f22.handle)

plt.figure()
plt.subplot(2,1,1)
plt.contourf(f222.X, f222.Y, f222.values)
plt.subplot(2,1,2)
plt.contourf(f222.X, f222.Y, f222.handle(f222.X, f222.Y))"""	

rect2 = Rectangle((0,0), (1,1), resol=7)
def u_D_term2(x, y):
	return np.logical_and(x >= 0.5, y <= 0.6)*2.0
u_D2 = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term2(x,y))

def boundary_D_boolean2(x): # special Dirichlet boundary condition
		if x[0] >= 0.6-tol and x[1] <= 0.5:
			return True
		elif x[0] <= 10**-8:
			return True
		else:
			return False

def u_D_term(x, y):
	return np.logical_and(x >= 80-tol, y <= 40)*2.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] >= 90-tol and x[1] <= 30:
			return True
		elif x[0] <= 10**-8 and x[1] >= 10 and x[1] <= 40:
			return True
		else:
			return False


"""def boundary_D_boolean(x):
	if (x[1] > 30-10**(-8) and x[1] < 50+10**(-8)) and (x[0] < 10**-8 or x[0] > 180-10**-8):
		return True
	else:
		return False"""

#u_D = mor.mapOnRectangle(rect, "handle", lambda x, y: np.logical_and(x < 10**-8, True)*1.0) #- 2.0*np.logical_and(x > 180-10**-8, True)*1.0)

fwd = linEllipt2dRectangle(rect, f2, u_D, boundary_D_boolean)
fwd2 = linEllipt2dRectangle(rect2, mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x), u_D2, boundary_D_boolean2)

"""k0 = mor.mapOnRectangle(rect, "handle", lambda x,y: 0*x + 1)
F0 = fwd.solve(k0)
F0_2 = fwd2.solve(k0)

plt.figure(); plt.ion()
plt.contourf(F0.X, F0.Y, F0.values, 60); plt.colorbar()
plt.show()

plt.figure(); plt.ion()
plt.contourf(F0_2.X, F0_2.Y, F0_2.values, 60); plt.colorbar()
plt.show()"""

m1 = GeneralizedGaussianWavelet2d(rect, 1.0, 1.5, 5)
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
plt.show()

"""temp = np.zeros((2**5,))
temp[12] = -1
u1 = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
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

