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
import smtplib
from email.mime.text import MIMEText

resol = 4

rect = Rectangle((0,0), (1,1), resol=resol)
rectFine = Rectangle((0,0), (1,1), resol=resol+1)
gamma = 0.005

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
		return  -1/log10(e)*np.logical_or(np.logical_and(np.logical_and(x < 0.5, x >= 0.4375), y < 0.5), np.logical_and(np.logical_and( x< 0.5 , x >= 0.4375) ,y >= 0.625 - tol)) + 2/log10(e)*np.logical_and(np.logical_and(x < 0.8125, x >= 0.75), np.logical_and(y >= 0.1875, y < 0.75)) + 0

def myUTruth2(x, y):
	return np.logical_and(np.logical_and((x-1)**2 + y**2 <= 0.65**2, (x-1)**2 + y**2 > 0.55**2), x<0.85)*-2/log10(e) + np.sin(4*x)*0.5/log10(e) - 4/log10(e)*np.logical_and(np.logical_and( x< 0.5 , x >= 0.4375) ,y >= 0.625 - tol) 
	
def u_D_term(x, y):
	return np.logical_and(x <= 0.1, True)*0.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))
u_D_fine = mor.mapOnRectangle(rectFine, "wavelet", parseResolution(u_D.waveletcoeffs, rectFine.resol))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		#if x[0] >= 0.6-tol and x[1] <= 0.5:
		#	return True
		#elif x[0] <= 10**-8:
		#	return True
		if x[0] < 1e-14:
			return True
		else:
			return False

f = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-.8)**2 + (y-.2)**2) < 0.2**2)*(20.0))
f_fine = mor.mapOnRectangle(rectFine, "wavelet", parseResolution(f.waveletcoeffs, rectFine.resol))
fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
fwd_fine = linEllipt2dRectangle(rectFine, f_fine, u_D_fine, boundary_D_boolean)

m1 = GeneralizedGaussianWavelet2d(rect, 0.01, 1.5, rect.resol+1)
m1_fine = GeneralizedGaussianWavelet2d(rectFine, 0.01, 1.5, rectFine.resol+1)

invProb = inverseProblem(fwd, m1, gamma)
invProb_fine = inverseProblem(fwd_fine, m1_fine, gamma)


N_obs = 50
N_obs = (int(sqrt(N_obs)))**2
obsposx = np.linspace(0.1, 0.9, int(sqrt(N_obs)))
obsposy = np.linspace(0.1, 0.9, int(sqrt(N_obs)))
xx, yy = np.meshgrid(obsposx, obsposy)
obspos = [xx.flatten(), yy.flatten()]
#obspos = np.random.uniform(0, 1, (2, N_obs))
#obspos = [obspos[0,:], obspos[1, :]]
invProb.obspos = obspos
invProb_fine.obspos = obspos

#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
#uTruth = m1.sample()
uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth2(x, y))

obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.plotSolAndLogPermeability(uTruth, obs=obs, save="ground_truth.png", blocky=True)
invProb.obs = obs
invProb_fine.obs = obs
#u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
#uOpt = invProb.find_uMAP(u0, nit=3,  method='BFGS', adjoint=False)
start=time.time()
uOpt_adj = invProb.find_uMAP(u0, nit=400, method = 'BFGS', adjoint=True, version=0)
#uOpt_adj = invProb.find_uMAP(uOpt_adj, nit=200, method = 'BFGS', adjoint=True, version=0)
end=time.time()
print("Optimization took " + str(end-start) + " seconds")
invProb.plotSolAndLogPermeability(uOpt_adj, obs=obs, blocky=True)

uu = mor.mapOnRectangle(rect, "wavelet", uOpt_adj.waveletcoeffs)
uu_fine = mor.mapOnRectangle(rectFine, "wavelet", parseResolution(uu.waveletcoeffs, rectFine.resol))

D_0 = invProb.DI_adjoint_vec_wavelet(uu, version=0)
D_2 = invProb.DI_adjoint_vec_wavelet(uu, version=2)

plt.figure();plt.plot(D_2)
plt.plot(D_0)



