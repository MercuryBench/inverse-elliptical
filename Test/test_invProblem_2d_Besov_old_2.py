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

np.random.seed(1877262)

rect = Rectangle((0,0), (1,1), resol=5)
gamma = 0.01
kappa = 25.0*2000**2
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
		return  0.2*(-4/log10(e)*np.logical_or(np.logical_and(np.logical_and(x <= 0.5 +tol, x >= 0.45 - tol), y <= 0.5+tol), np.logical_and(np.logical_and( x<= 0.5+tol , x >= 0.45 - tol) ,y >= 0.6 - tol)) + 2/log10(e)*np.logical_and(np.logical_and(x <= 0.75 + tol, x >= 0.7 - tol), np.logical_and(y >= 0.2 - tol, y <= 0.8+tol)) + 0)

def myUTruth2(x, y):
	return np.logical_and(np.logical_and((x-1)**2 + y**2 <= 0.65**2, (x-1)**2 + y**2 > 0.55**2), x<0.85)*-2/log10(e) + np.sin(4*x)*0.5/log10(e) - 4/log10(e)*np.logical_and(np.logical_and( x< 0.5 , x >= 0.4375) ,y >= 0.625 - tol) 

def u_D_term(x, y):
	return np.logical_and(x >= 1.0-1e-10, True)*2.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] >= 1.0-tol:
			return True
		elif x[0] <= tol:
			return True
		else:
			return False

f = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
m1 = Besov11Wavelet(rect, kappa, 1.0, 5)
invProb = inverseProblem(fwd, m1, gamma)


obspos = np.random.uniform(0, 1, (2, N_obs))
obspos = [obspos[0,:], obspos[1, :]]
invProb.obspos = obspos

uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth2(x, y))
obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.obs = obs
invProb.plotSolAndLogPermeability(uTruth, obs=obs, blocky=True)


N_opt = 100

uOpt2 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
tau = 0.0000001

def shrinkage(z, steptau):
	retvals = np.zeros((len(z),))
	#cutoff = 2*tau*gamma**2*sqrt(kappa)*5000
	cutoff = 2*steptau*gamma**2*sqrt(kappa)
	for k in range(len(z)):
		if z[k] >= cutoff:
			retvals[k] = z[k]-cutoff
		elif z[k] >= -cutoff:
			retvals[k] = 0
		else: 
			retvals[k] = z[k]+cutoff
	return retvals

valI = np.zeros(N_opt)
valPhi = np.zeros(N_opt)
valnorm = np.zeros(N_opt)
valI2 = np.zeros(N_opt)
valPhi2 = np.zeros(N_opt)
valnorm2 = np.zeros(N_opt)
valI3 = np.zeros(N_opt)
valPhi3 = np.zeros(N_opt)
valnorm3 = np.zeros(N_opt)
valI4 = np.zeros(N_opt)
valPhi4 = np.zeros(N_opt)


def makestep(u0, direction, tau):
	return mor.mapOnRectangle(invProb.rect, "wavelet", packWavelet(unpackWavelet(u0.waveletcoeffs) - tau*direction))

def shrinkFnc(u0, steptau):
	return mor.mapOnRectangle(invProb.rect, "wavelet", packWavelet(shrinkage(unpackWavelet(u0.waveletcoeffs), steptau)))

"""def linesearch(uk, direction, rho, alpha):
	uTemp = mor.mapOnRectangle(invProb.rect, "wavelet", packWavelet(unpackWavelet(uk.waveletcoeffs) - tau_init*unpackWavelet(direction.waveletcoeffs)))
	while (invProb.Phi(uTemp) > invProb.Phi(uk) - alpha*np.dot(unpackWavelet(direction.waveletcoeffs).T, unpackWavelet(direction.waveletcoeffs)):
		alpha = alpha*rho
	return alpha"""

"""def backtracking(f, pos, Dfpos, direction, c=0.1, alpha=1.0, tau=0.3, debug=False): # tau: shrinking param, c: minimal decline threshold, alpha: initial step size
	m = np.dot(direction.T, Dfpos)
	if debug:
		print("local slope: " + str(m))
	assert(m < 0)	
	for j in range(1000):
		if debug:
			print("---")
			print("step " + str(j))
		t = -c*m
		if debug:		
			#print("newpos = " + str(pos + alpha*direction))
			print("f(pos) = " + str(f(pos)))
			print("f(newpos) = " + str(f(pos + alpha*direction)))
			print("f(pos) - f(newpos) = " + str(f(pos) - f(pos + alpha*direction)) + " and must > than " + str(alpha*t))
		if np.isnan(f(pos + alpha*direction)) or f(pos) - f(pos + alpha*direction) < alpha*t:
			if debug:
				print("is not, hence make smaller")
			alpha = tau*alpha
		else:
			if debug:
				print("alright, this is it")
			return alpha
	raise Exception("damn")
	

def linesearch(uk, direction):
	f_ = lambda wc: invProb.Phi(mor.mapOnRectangle(invProb.rect, "wavelet", packWavelet(wc)))
	pos_ = unpackWavelet(uk.waveletcoeffs)
	Dfpos_ = unpackWavelet(direction.waveletcoeffs)
	direction_ = -unpackWavelet(direction.waveletcoeffs)
	return backtracking(f_, pos_, Dfpos_, direction_, tau = 0.1, debug=False)"""


# ISTA with line search
uOpt = mor.mapOnRectangle(rect, "wavelet", m1._mean)
valI[0] = invProb.I(uOpt)
valnorm[0] = invProb.prior.normpart(uOpt)
valPhi[0] = invProb.Phi(uOpt)
alpha = tau*100

c = 0.3

for l in range(N_opt-1):
	alpha = tau*100
	DPhi = invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(invProb.rect, "wavelet", uOpt.waveletcoeffs), version=0)
	temp = shrinkage(unpackWavelet(uOpt.waveletcoeffs) - alpha*DPhi, alpha)
	proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
	maxnum = 20
	
	failed = False
	
	while np.isnan(invProb.I(proposal)) or invProb.I(proposal) > invProb.I(uOpt) or invProb.I(proposal) > invProb.Phi(uOpt) + np.dot(DPhi.T, temp-unpackWavelet(uOpt.waveletcoeffs)) + 1/(c*2.0*alpha)*np.dot((temp-unpackWavelet(uOpt.waveletcoeffs)).T, temp-unpackWavelet(uOpt.waveletcoeffs)) + invProb.prior.normpart(uOpt):
		alpha = alpha/3.0
		temp = shrinkage(unpackWavelet(uOpt.waveletcoeffs) - alpha*DPhi, alpha)
		proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
		maxnum -= 1
		if maxnum == 0:
			print("too much backtracking, try without proper descent")
			failed = True
			break
			
	alpha = tau*100
	temp = shrinkage(unpackWavelet(uOpt.waveletcoeffs) - alpha*DPhi, alpha)
	proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
	
	while np.isnan(invProb.I(proposal)) or invProb.I(proposal) > invProb.Phi(uOpt) + np.dot(DPhi.T, temp-unpackWavelet(uOpt.waveletcoeffs)) + 1/(c*2.0*alpha)*np.dot((temp-unpackWavelet(uOpt.waveletcoeffs)).T, temp-unpackWavelet(uOpt.waveletcoeffs)) + invProb.prior.normpart(uOpt):
		alpha = alpha/3.0
		temp = shrinkage(unpackWavelet(uOpt.waveletcoeffs) - alpha*DPhi, alpha)
		proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
		maxnum -= 1
		if maxnum == 0:
			print("something is seriously wrong!")
			break
	uOpt = proposal
	valI[l+1] = invProb.I(uOpt)
	valnorm[l+1] = invProb.prior.normpart(uOpt)
	valPhi[l+1] = invProb.Phi(uOpt)


# ISTA without line search

uOpt2 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
valI2[0] = invProb.I(uOpt2)
valnorm2[0] = invProb.prior.normpart(uOpt2)
valPhi2[0] = invProb.Phi(uOpt2)
for l in range(N_opt-1):
	DPhi = invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(invProb.rect, "wavelet", uOpt2.waveletcoeffs), version=0)
	# uTemp = uOpt2 - mor.mapOnRectangle(rect, "wavelet", packWaveletDPhi)*tau
	temp = shrinkage(unpackWavelet(uOpt2.waveletcoeffs) - tau*DPhi, tau)
	uOpt2 = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
	valI2[l+1] = invProb.I(uOpt2)
	valnorm2[l+1] = invProb.prior.normpart(uOpt2)
	valPhi2[l+1] = invProb.Phi(uOpt2)


xk = unpackWavelet(m1._mean)
xkm1 = unpackWavelet(m1._mean)
yk = unpackWavelet(m1._mean)
ykp1 = unpackWavelet(m1._mean)
tks = np.zeros((N_opt,))
tks[1] = 1
valI3[0] = invProb.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(xk)))
valPhi3[0] = invProb.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(xk)))
valnorm3[0] = invProb.prior.normpart(mor.mapOnRectangle(rect, "wavelet", packWavelet(xk)))

countrestarts = 0;

# FISTA with line search

"""
for l in range(N_opt-1):
	alpha = tau*100
	DPhi = invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(invProb.rect, "wavelet", uOpt.waveletcoeffs), version=0)
	temp = shrinkage(unpackWavelet(uOpt.waveletcoeffs) - alpha*DPhi, alpha)
	proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
	maxnum = 20
	
	failed = False
	
	while np.isnan(invProb.I(proposal)) or invProb.I(proposal) > invProb.I(uOpt) or invProb.I(proposal) > invProb.Phi(uOpt) + np.dot(DPhi.T, temp-unpackWavelet(uOpt.waveletcoeffs)) + 1/(c*2.0*alpha)*np.dot((temp-unpackWavelet(uOpt.waveletcoeffs)).T, temp-unpackWavelet(uOpt.waveletcoeffs)) + invProb.prior.normpart(uOpt):
		alpha = alpha/3.0
		temp = shrinkage(unpackWavelet(uOpt.waveletcoeffs) - alpha*DPhi, alpha)
		proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
		maxnum -= 1
		if maxnum == 0:
			print("too much backtracking, try without proper descent")
			failed = True
			break
			
	alpha = tau*100
	temp = shrinkage(unpackWavelet(uOpt.waveletcoeffs) - alpha*DPhi, alpha)
	proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
	
	while np.isnan(invProb.I(proposal)) or invProb.I(proposal) > invProb.Phi(uOpt) + np.dot(DPhi.T, temp-unpackWavelet(uOpt.waveletcoeffs)) + 1/(c*2.0*alpha)*np.dot((temp-unpackWavelet(uOpt.waveletcoeffs)).T, temp-unpackWavelet(uOpt.waveletcoeffs)) + invProb.prior.normpart(uOpt):
		alpha = alpha/3.0
		temp = shrinkage(unpackWavelet(uOpt.waveletcoeffs) - alpha*DPhi, alpha)
		proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
		maxnum -= 1
		if maxnum == 0:
			print("something is seriously wrong!")
			break
	uOpt = proposal
	valI[l+1] = invProb.I(uOpt)
	valnorm[l+1] = invProb.prior.normpart(uOpt)
	valPhi[l+1] = invProb.Phi(uOpt)
"""


for l in range(1, N_opt-1):
	alpha = tau*100
	yk_fnc = mor.mapOnRectangle(invProb.rect, "wavelet", packWavelet(yk))
	DPhi = invProb.DPhi_adjoint_vec_wavelet(yk_fnc, version=0)
	temp = shrinkage(unpackWavelet(yk_fnc.waveletcoeffs) - alpha*DPhi, alpha)
	proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
	
	uOpt3 = mor.mapOnRectangle(rect, "wavelet", packWavelet(xk))
	maxnum = 20
	failed = False
	while np.isnan(invProb.I(proposal)) or invProb.I(proposal) > invProb.I(yk_fnc) or invProb.I(proposal) > invProb.Phi(yk_fnc) + np.dot(DPhi.T, temp-unpackWavelet(yk_fnc.waveletcoeffs)) + 1/(c*2.0*alpha)*np.dot((temp-unpackWavelet(yk_fnc.waveletcoeffs)).T, temp-unpackWavelet(yk_fnc.waveletcoeffs)) + invProb.prior.normpart(yk_fnc):
		alpha = alpha/3.0
		temp = shrinkage(unpackWavelet(yk_fnc.waveletcoeffs) - alpha*DPhi, alpha)
		proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
		maxnum -= 1
		
		if maxnum == 0:
			print("too much backtracking (in FISTA), try without proper descent")
			failed = True
			break
	maxnum = 20
	failed = False
	while np.isnan(invProb.I(proposal)) or invProb.I(proposal) > invProb.Phi(yk_fnc) + np.dot(DPhi.T, temp-unpackWavelet(yk_fnc.waveletcoeffs)) + 1/(c*2.0*alpha)*np.dot((temp-unpackWavelet(yk_fnc.waveletcoeffs)).T, temp-unpackWavelet(yk_fnc.waveletcoeffs)) + invProb.prior.normpart(yk_fnc):
		alpha = alpha/3.0
		temp = shrinkage(unpackWavelet(uOpt3.waveletcoeffs) - alpha*DPhi, alpha)
		proposal = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))
		maxnum -= 1
		
		if maxnum == 0:
			print("something is seriously wrong! (in FISTA)")
			break
	xk = shrinkage(yk - alpha*DPhi, alpha)
	uOpt3 = mor.mapOnRectangle(invProb.rect, "wavelet", packWavelet(xk))
	valI3[l+1] = invProb.I(uOpt3)
	valnorm3[l+1] = invProb.prior.normpart(uOpt3)
	valPhi3[l+1] = invProb.Phi(uOpt3)
	
	if (l <= N_opt-2):
		tks[l+1] = 0.5 * (1.0 + sqrt(1.0 + 4.0*tks[l]**2))
		ykp1 = xk + (tks[l]-1)/tks[l+1] * (xk - xkm1)
	
	if valI[l] > valI[l-1] and l <= N_opt-2:
		# restart
		tks[l+1] = 1
		ykp1 = xk
		countrestarts += 1
	
	if l <= N_opt-2:
		yk = ykp1
		xkm1 = xk

print("#restarts = " + str(countrestarts))

xk_ = unpackWavelet(m1._mean)
xkm1_ = unpackWavelet(m1._mean)
yk_ = unpackWavelet(m1._mean)
ykp1_ = unpackWavelet(m1._mean)
tks_ = np.zeros((N_opt,))
tks_[1] = 1

valI4[0] = invProb.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(xk_)))
valPhi4[0] = invProb.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(xk_)))

countrestarts = 0
# FISTA without line search

for l in range(1, N_opt):
	DPhi = invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(invProb.rect, "wavelet", packWavelet(yk_)), version=0)
	# uTemp = uOpt2 - mor.mapOnRectangle(rect, "wavelet", packWaveletDPhi)*tau
	#alpha = linesearch(mor.mapOnRectangle(invProb.rect, "wavelet", packWavelet(yk_)), mor.mapOnRectangle(invProb.rect, "wavelet", packWavelet(DPhi)))
	xk_ = shrinkage(yk_ - tau*DPhi, tau)
	if (l <= N_opt-2):
		tks_[l+1] = 0.5 * (1.0 + sqrt(1.0 + 4.0*tks_[l]**2))
		ykp1_ = xk_ + (tks_[l]-1)/tks_[l+1] * (xk_ - xkm1_)
	uOpt4 = mor.mapOnRectangle(rect, "wavelet", packWavelet(xk_))
	valI4[l] = invProb.I(uOpt4)
	valPhi4[l] = invProb.Phi(uOpt4)
	
	if valI4[l] > valI4[l-1] and l <= N_opt-2:
		# restart
		tks_[l+1] = 1
		ykp1_ = xk_
		countrestarts += 1
	
	if l <= N_opt-2:
		yk_ = ykp1_
		xkm1_ = xk_


print("#restarts = " + str(countrestarts))

"""ts = np.linspace(0, tau, 100)
newVals = np.zeros((100,))
newValsShrunk = np.zeros((100,))
DPhi = invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(invProb.rect, "wavelet", uOpt2.waveletcoeffs), version=0)
for n, t in enumerate(ts):
	s = makestep(uOpt2, DPhi, t)
	newVals[n] = invProb.I(s)
	newValsShrunk[n] = invProb.I(shrinkFnc(s))"""

"""plt.figure();
plt.plot(ts, newVals, 'r')
plt.plot(ts, newValsShrunk, 'g')
plt.plot(ts, invProb.I(uOpt2)*np.ones((100,)), 'k--')"""

plt.figure()
plt.subplot(311)
plt.semilogy(valI2, 'g.--', label="I_ISTA_")
plt.semilogy(valI, 'r.--', label="I_ISTA_backtr")
#plt.semilogy(valI4, 'm--', label="I_FISTA");
plt.semilogy(valI3, 'k--', label="I_FISTA_backtr");
plt.legend()
plt.subplot(312)
plt.semilogy(valPhi, 'r.-', label="Phi_ISTA_backtr")
plt.semilogy(valPhi2, 'g.-', label="Phi_ISTA")
plt.semilogy(valPhi3, 'k', label="Phi_FISTA_backtr")
#plt.semilogy(valPhi4, 'm', label="Phi_FISTA")
plt.legend()
plt.subplot(313)
#plt.semilogy(valPhi, 'k', label="Phi_FISTA_backtr")
plt.semilogy(valnorm, 'r.-', label="norm_ISTA_backtr")
plt.semilogy(valnorm2, 'g.-', label="norm_ISTA")
plt.semilogy(valnorm3, 'k.-', label="norm_FISTA_backtr")
#plt.semilogy(valPhi4, 'm', label="Phi_FISTA")
plt.legend()


invProb.plotSolAndLogPermeability(uOpt, obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(uOpt2, obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(uOpt3, obs=obs, blocky=True)
#invProb.plotSolAndLogPermeability(uOpt4, obs=obs, blocky=True)

"""plt.figure();
plt.plot(unpackWavelet(uOpt2.waveletcoeffs), 'g.-');
plt.plot(unpackWavelet(uOpt.waveletcoeffs), 'k.-');
plt.plot(unpackWavelet(uOpt3.waveletcoeffs), 'r.-');
plt.plot(unpackWavelet(uOpt4.waveletcoeffs), 'm.-');"""




print("squared Cameron-Martin norm of ground truth: " + str(invProb.prior.normpart(uTruth)))
print("squared misfit of ground Truth: " + str(invProb.Phi(uTruth, obs)))

print("squared Cameron-Martin norm of linesearch-ISTA optimizer: " + str(invProb.prior.normpart(uOpt)))
print("squared misfit of linesearch-ISTA optimizer: " + str(invProb.Phi(uOpt, obs)))
print("squared Cameron-Martin norm of ISTA optimizer: " + str(invProb.prior.normpart(uOpt2)))
print("squared misfit of ISTA optimizer: " + str(invProb.Phi(uOpt2, obs)))
print("squared Cameron-Martin norm of backtrack-FISTA optimizer: " + str(invProb.prior.normpart(uOpt3)))
print("squared misfit of backtrack-FISTA optimizer: " + str(invProb.Phi(uOpt3, obs)))
"""print("squared Cameron-Martin norm of FISTA optimizer: " + str(invProb.prior.normpart(uOpt4)))
print("squared misfit of FISTA optimizer: " + str(invProb.Phi(uOpt4, obs)))"""

#print("squared Cameron-Martin norm of truncated ground truth: " + str(invProb.prior.normpart(uTruth_trunc)))
#print("squared misfit of truncated ground Truth: " + str(invProb.Phi(uTruth_trunc, obs)))




