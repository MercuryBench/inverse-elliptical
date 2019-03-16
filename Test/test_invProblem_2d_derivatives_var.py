from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
import math
import sys 
sys.path.append('..')
import mapOnRectangle as mor
from fwdProblem import *
from invProblem2d import *
from rectangle import *

rect = Rectangle((0,0), (1,1), resol=6)
gamma = 0.01

N_obs = 100

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
		return  -4/log10(e)*np.logical_or(np.logical_and(np.logical_and(x <= 0.5 +tol, x >= 0.45 - tol), y <= 0.5+tol), np.logical_and(np.logical_and( x<= 0.5+tol , x >= 0.45 - tol) ,y >= 0.6 - tol)) + 2/log10(e)*np.logical_and(np.logical_and(x <= 0.75 + tol, x >= 0.7 - tol), np.logical_and(y >= 0.2 - tol, y <= 0.8+tol)) + 0

def u_D_term(x, y):
	return np.logical_and(x >= 0.5, y <= 0.6)*2.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] >= 0.6-tol and x[1] <= 0.5:
			return True
		elif x[0] <= 10**-8:
			return True
		else:
			return False

f = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
m1 = GeneralizedGaussianWavelet2d(rect, 1, 0.5, 5)
invProb = inverseProblem(fwd, m1, gamma)

uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth(x, y))
obspos = np.random.uniform(0, 1, (2, N_obs))
obspos = [obspos[0,:], obspos[1, :]]
invProb.obspos = obspos
obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.plotSolAndLogPermeability(uTruth)

Nmodes = len(unpackWavelet(uTruth.waveletcoeffs))
temp = np.zeros((Nmodes,))
temp[17] = 0.1
h = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))

u = uTruth
v = u + h
invProb.plotSolAndLogPermeability(v, obs=obs)

Du_h = invProb.DFfnc(u, h)

Fu = invProb.Ffnc(u)
Fv = invProb.Ffnc(v)


D2u_h = invProb.D2Ffnc(u, h, h)

# optical check for approximation in observation space (Fu, Fu+DFu(h), Fu+DFu(h)+0.5*D2Fu[h,h], Fv) where v = u + h

plt.figure();
plt.subplot(4, 1, 1)
plt.contourf(Fu.values)
plt.subplot(4, 1, 2)
plt.contourf(Fu.values + Du_h.values)
plt.subplot(4, 1, 3)
plt.contourf(Fu.values + Du_h.values + 0.5*D2u_h.values)
plt.subplot(4, 1, 4)
plt.contourf(Fv.values)

# computational check

print("norm Fu-Fv: " + str(np.sum((Fu.values-Fv.values)**2)))

print("norm Fu+DFu-Fv: " + str(np.sum((Fu.values+Du_h.values-Fv.values)**2)))

print("norm Fu+DFu+0.5*D2Fu-Fv: " + str(np.sum((Fu.values+Du_h.values+0.5*D2u_h.values-Fv.values)**2)))

# optical check for approximation of I(u): (I(u), I(u)+DI(u)(h), I(u)+DI(u)(h)+D2I(u)[h,h], DI(v))
#wtilde = fwd.solveWithDiracRHS(k, ws, xs, pureFenicsOutput=False)
hfactor = np.linspace(-1, 1, 101)
invProb.obs = obs
Ivals = np.array([invProb.I(u+h*hfac) for hfac in hfactor])
Ivals_D = invProb.I(u) + hfactor*invProb.DI(u, h)
Ivals_D2 = invProb.I(u) + hfactor*invProb.DI(u, h) + 0.5*hfactor**2*invProb.D2I(u, h, h)

plt.figure();
plt.plot(hfactor, Ivals, '.-')

plt.plot(hfactor, Ivals_D, 'r.-')
plt.plot(hfactor, Ivals_D2, 'g.-')



#DD = invProb.gradientI_adjoint_wavelet(u)
invProb.DPhi(u, h)
discrepancy = obs-invProb.Gfnc(u, obspos=obspos)
DG_of_u_h = invProb.DGfnc(u, h, obspos=obspos)	
value0 = -1.0/(gamma**2)*np.dot(discrepancy, DG_of_u_h)


# primal method
k = mor.mapOnRectangle(rect, "expl", np.exp(u.values))
k1 = k*h
y = Fu
w = fwd.solveWithHminus1RHS(k, k1, y, pureFenicsOutput=False) # solves -div(k*nabla(y1)) = div(k1*nabla(y)) for y1	
value1 = -1/gamma**2 * sum((obs-Fu.handle(obspos[0], obspos[1]))*w.handle(obspos[0], obspos[1]))

#dual method
ws = obs - Fu.handle(obspos[0], obspos[1])
xs = zip(obspos[0], obspos[1])
wtilde_ = fwd.solveWithDiracRHS(k, ws, xs, pureFenicsOutput=True)
wtilde = fwd.solveWithDiracRHS(k, ws, xs, pureFenicsOutput=False) # solves -div(k*nabla(y)) = sum_i w_i*dirac_{x_i}
Fu_ = invProb.Ffnc(u, pureFenicsOutput=True)
value2 = assemble(morToFenicsConverter(k1, fwd.mesh, fwd.V)*dot(grad(Fu_), grad(wtilde_))*dx)


