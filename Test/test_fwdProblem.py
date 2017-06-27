from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from fwdProblem import *
from measures import *



rect = Rectangle((0,0), (180,78), resol=7)

f2 = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-40)**2 + (y-20)**2) < 4)*(-8.2667/100))
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
		
def boundary_D_boolean(x):
	if x[1] > 10**(-8):
		return True
	else:
		return False

u_D = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)

fwd = linEllipt2dRectangle(rect, f2, u_D, boundary_D_boolean)
k0 = mor.mapOnRectangle(rect, "handle", lambda x,y: 0*x + 1)
F0 = fwd.solve(k0)

plt.figure(); plt.ion()
plt.contourf(F0.X, F0.Y, F0.values, 30); plt.colorbar()
plt.show()

m1 = GeneralizedGaussianWavelet2d(rect, 1.0, 0.5, 5)
u1 = m1.sample()
k1 = mor.mapOnRectangle(rect, "handle", lambda x,y: np.exp(u1.handle(x, y)))
F1_ = fwd.solve(k1, pureFenicsOutput=True)
F1 = fwd.solve(k1)
vtkfile = File('solution_sandbox.pvd')
vtkfile << F1_

plt.figure(); plt.ion()
plt.contourf(F1.X, F1.Y, F1.values, 30); plt.colorbar()
plt.show()

