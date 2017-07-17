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
from fenics import *
from measures import *

rect = Rectangle((0,0), (180,78), resol=5)
gamma = 0.001

N_obs = 100

u_D = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)

def boundary_D_boolean(x):
	if x[1] > 10**(-8):
		return True
	else:
		return False

f = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-40)**2 + (y-20)**2) < 4)*(-.01))

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
prior = GeneralizedGaussianWavelet2d(rect, 0.1, 0.0, 3)
invProb = inverseProblem(fwd, prior, gamma)


obspos = [np.random.uniform(0, 180, N_obs), np.random.uniform(0, 78, N_obs)]

invProb.obspos = obspos

uTruth = prior.sample()

obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.obs = obs
invProb.plotSolAndLogPermeability(uTruth, obs=obs, obspos=obspos)

wc_unpacked = unpackWavelet(uTruth.waveletcoeffs)

assert(invProb.I(uTruth, obs) == invProb.I_forOpt(wc_unpacked))
print(invProb.I(uTruth, obs))
print(invProb.Phi(uTruth, obs))

u0 = mor.mapOnRectangle(rect, "wavelet", prior._mean)

print(invProb.I_forOpt(wc_unpacked))
uOpt = invProb.find_uMAP(u0)

