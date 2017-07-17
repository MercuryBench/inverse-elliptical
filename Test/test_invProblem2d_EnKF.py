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
def myUTruth3(x,y):
	return 1 - 4.0*np.logical_and(np.logical_and(x >= 0.375, x < 0.75), y < 0.625)
def myUTruth2(x,y):
	return 1.0 - 2.0*np.logical_or(np.logical_and(np.logical_and(x>=0.25, x<0.75), y < 0.5), np.logical_and(np.logical_and(x>=0.25, x<0.75), y >= 0.75))
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

f = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)# (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
m1 = GeneralizedGaussianWavelet2d(rect, 0.01, 0.5, 5)
invProb = inverseProblem(fwd, m1, gamma)

N_obs = 100
obspos = np.random.uniform(0, 1, (2, N_obs))
obspos = [obspos[0,:], obspos[1, :]]
invProb.obspos = obspos

def unitvec(N, k):
	temp = np.zeros((N,))
	temp[k] = 1.0
	return temp

#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
#uTruth = m1.sample()
uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth3(x, y))
obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.obs = obs
invProb.plotSolAndLogPermeability(uTruth, obs=obs)

#u_new, u_new_mean, us = invProb.EnKF(obs, 128, KL=False, N = 1)
#invProb.plotSolAndLogPermeability(u_new_mean)
print("0.01")
u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
#uOpt = invProb.find_uMAP(u0, nit=100, nfev=100)
uOptList = [u0]
for k in range(5):
	uOptList.append(invProb.find_uMAP(uOptList[-1], nit=400, nfev=100, method='BFGS'))
	invProb.plotSolAndLogPermeability(uOptList[-1])
uOptList = uOptList[1:-1]
Phis = [invProb.Phi(uO) for uO in uOptList]
priorparts = [invProb.prior.normpart(uO) for uO in uOptList]
Is = [invProb.I(uO) for uO in uOptList]

plt.figure();
plt.plot(Is, 'b')
plt.plot(Phis, 'g')
plt.plot(priorparts, 'r')


#uList, uListUnique, PhiList = invProb.randomwalk_pCN(u0, 1000)
#u_new, u_new_mean, us, vals, vals_mean = invProb.EnKF(obs, 50, KL=False, N = 10, beta = 0.05)
#plt.figure(); plt.semilogy(PhiList)
#invProb.plotSolAndLogPermeability(uList[-1])
"""plt.figure();
plt.semilogy(vals)
plt.semilogy(vals_mean, 'r', linewidth=4)
invProb.plotSolAndLogPermeability(u_new_mean)"""


"""Is = []
hvals = np.linspace(-1,1,200)

for k in range(16):
	d = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(16, k)))
	Is.append([invProb.I(u_new_mean + d*h) for h in hvals])

plt.figure()
for k in range(16):
	plt.subplot(4,4,k+1)
	plt.plot(hvals, Is[k])"""
