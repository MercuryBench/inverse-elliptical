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
from fista import *
import scipy.optimize as opt

rect = Rectangle((0,0), (1,1), resol=6)
gamma = 1

np.random.seed(7)

def uTruth_handle(x, y):
	return np.logical_and(np.logical_and((x-1)**2 + y**2 <= 0.65**2, (x-1)**2 + y**2 > 0.55**2), x<0.85)*-2/log10(e) + np.sin(4*x)*0.5/log10(e) - 4/log10(e)*np.logical_and(np.logical_and( x< 0.5 , x >= 0.4375) ,y >= 0.625 - tol) 
	
def u_D_term(x, y):
	return np.logical_and(True, y <= 1e-8)*0.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] <= 10**-8:
			return True
		else:
			return False

f = mor.mapOnRectangle(rect, "handle", lambda x, y: ( ((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-2000.0) + (( (x-.2)**2 + (y-.75)**2) < 0.1**2)*2000.0 + (( (x-.8)**2 + (y-.2)**2) < 0.1**2)*2000.0)

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)

kappavals = [1.0, 10.0, 100.0]
alreadyShown = False
kappa = 1.0

for kappa in kappavals:
	m1 = Besov11Wavelet(rect, kappa, 1.0, 6) 
	invProb = inverseProblem(fwd, m1, gamma)
	uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: uTruth_handle(x, y))
		


	N_obs = 49
	N_iter = 4000
	obspos = np.random.uniform(0, 1, (2, N_obs))
	obspos = [obspos[0,:], obspos[1, :]]
	obsposx = np.linspace(0.05, 0.95, 7)
	obsposy = np.linspace(0.05, 0.95, 7)
	OX, OY = np.meshgrid(obsposx, obsposy)
	obspos = np.stack((OX, OY)).reshape((2, 49))
	obspos = [obspos[0,:], obspos[1, :]]
	invProb.obspos = obspos


	obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
	invProb.obs = obs
	if not(alreadyShown):
		invProb.plotSolAndLogPermeability(uTruth, obs=obs, save="ground_truth.pdf", blocky=True)
		alreadyShown = True
	u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)

	tau = 0.000005


	x0 = unpackWavelet(m1._mean)
	alpha0 = tau
	I_fnc = lambda x: invProb.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
	Phi_fnc = lambda x:  invProb.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
	DPhi_fnc = lambda x: np.array(invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)), version=0))
	DNormpart_fnc = lambda x: invProb.DNormpart(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
	DI_fnc = lambda x: DPhi_fnc(x) + DNormpart_fnc(x)

	"""
	scale = 0.0000001

	I_fnc_scaled = lambda x: I_fnc(scale*x)
	Phi_fnc_scaled = lambda x: Phi_fnc(scale*x)
	DPhi_fnc_scaled = lambda x: scale*DPhi_fnc(scale*x)
	DNormpart_fnc_scaled = lambda x: scale*DNormpart_fnc(scale*x)
	DI_fnc_scaled = lambda x: scale*DI_fnc(scale*x)

	result2 = opt.minimize(I_fnc_scaled, x0, method="CG", jac=DI_fnc_scaled, options={"maxiter": 20})
	xOpt_CG = result2["x"]*scale
	uOpt_CG = mor.mapOnRectangle(rect, "wavelet", packWavelet(xOpt_CG))"""
	cutoffmultiplier = 2*gamma**2 * sqrt(kappa)
	result = FISTA(x0, I_fnc, Phi_fnc, DPhi_fnc, cutoffmultiplier, alpha0=1.0, eta=0.5, N_iter=N_iter, backtracking=True, c=1.0, showDetails=True)
	xk, Is, Phis, num_bt = result["xk"], result["Is"], result["Phis"], result["num_backtrackings"]
	uOpt = mor.mapOnRectangle(rect, "wavelet", packWavelet(xk[-1]))

	invProb.plotSolAndLogPermeability(uOpt, obs=obs, save="uMAP_Besov_kappa=" + str(kappa) + ".pdf", blocky=True)


	plt.figure(); plt.ion();
	plt.subplot(311)
	plt.title("kappa = " + str(kappa))
	plt.plot(Is, label = "I")
	plt.legend()
	plt.subplot(312)
	plt.plot(Phis, label = "Phi")
	plt.legend()
	plt.subplot(313)
	plt.plot(Is-Phis, label = "norm")
	plt.legend()

"""kappa = 10
m2 = Besov11Wavelet(rect, kappa, 1.0, 5) # resol = 5
invProb2 = inverseProblem(fwd, m2, gamma)


invProb2.obspos = obspos
invProb2.obs = obs


x0 = unpackWavelet(m1._mean)
I_fnc = lambda x: invProb2.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
Phi_fnc = lambda x:  invProb2.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
DPhi_fnc = lambda x: np.array(invProb2.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)), version=0))
cutoffmultiplier = 2*gamma**2 * sqrt(kappa) #sqrt(kappa)
alpha0 = tau
N_iter = 100 #100

start = time.time()
result = FISTA(x0, I_fnc, Phi_fnc, DPhi_fnc, cutoffmultiplier, alpha0=alpha0, eta=0.5, N_iter=N_iter, backtracking=True, c=1.0, showDetails=True)
end = time.time()

xk, Is, Phis, num_bt = result["xk"], result["Is"], result["Phis"], result["num_backtrackings"]
uOpt = mor.mapOnRectangle(rect, "wavelet", packWavelet(xk[-1]))

invProb.plotSolAndLogPermeability(uOpt, obs=obs, blocky=True)


plt.figure(); plt.ion();
plt.subplot(311)
plt.plot(Is, label = "I")
plt.legend()
plt.subplot(312)
plt.plot(Phis, label = "Phi")
plt.legend()
plt.subplot(313)
plt.plot(Is-Phis, label = "norm")
plt.legend()"""



# Test gradient:
"""
ustar = uOpt_adj
direction = mor.mapOnRectangle(rect, "wavelet", packWavelet(np.random.normal(0, 0.0005, unpackWavelet(ustar.waveletcoeffs).shape)))
rs = np.linspace(-1, 1, 100)
Phiu = invProb.Phi(ustar)
DPhiu = invProb.DPhi_adjoint_vec_wavelet(ustar, version=0)
DPhiu_2 = invProb.DPhi_adjoint_vec_wavelet(ustar, version=2)
Phis = np.zeros(rs.shape)
PhisLin = np.zeros(rs.shape)
PhisLin_2 = np.zeros(rs.shape)
for n, r in enumerate(rs):
	Phis[n] = invProb.Phi(ustar + direction*r)
	PhisLin[n] = Phiu + np.dot(DPhiu, unpackWavelet(direction.waveletcoeffs)*r)
	PhisLin_2[n] = Phiu + np.dot(DPhiu_2, unpackWavelet(direction.waveletcoeffs)*r)

plt.figure();
plt.plot(rs, Phis, 'b')
plt.plot(rs, PhisLin, 'r')
plt.plot(rs, PhisLin_2, 'g')"""


