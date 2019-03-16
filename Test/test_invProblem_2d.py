from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
import math
from math import e
import sys 
sys.path.append('..')
import mapOnRectangle as mor
from fwdProblem import *
from invProblem2d import *
from rectangle import *

rect = Rectangle((0,0), (1,1), resol=5)
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
m1 = GeneralizedGaussianWavelet2d(rect, 0.001, 0.5, 5)
invProb = inverseProblem(fwd, m1, gamma)


obspos = np.random.uniform(0, 1, (2, N_obs))
obspos = [obspos[0,:], obspos[1, :]]
invProb.obspos = obspos

uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth(x, y))
obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.obs = obs
invProb.plotSolAndLogPermeability(uTruth, obs=obs)

uTruth_trunc = mor.mapOnRectangle(rect, "wavelet", uTruth.waveletcoeffs[0:m1.maxJ])
invProb.plotSolAndLogPermeability(uTruth_trunc, obs=obs)

print("squared Cameron-Martin norm of ground truth: " + str(invProb.prior.normpart(uTruth)))
print("squared misfit of ground Truth: " + str(invProb.Phi(uTruth, obs)))

print("squared Cameron-Martin norm of truncated ground truth: " + str(invProb.prior.normpart(uTruth_trunc)))
print("squared misfit of truncated ground Truth: " + str(invProb.Phi(uTruth_trunc, obs)))

import time
start = time.time();invProb.Ffnc(uTruth);end = time.time()
print("solve time: " + str(end-start))


u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
uOpt = invProb.find_uMAP(u0, nit=1000, nfev=1000, method='Nelder-Mead')
#uOpt = invProb.find_uMAP(u0, nit=10, nfev=100, method='CG')



def unitvec(N, k):
	temp = np.zeros((N,))
	temp[k] = 1.0
	return temp

