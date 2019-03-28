from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
import math
from math import e
import sys 
sys.path.append('../../..')
sys.path.append('../../../../RTO')
import mapOnRectangle as mor
from fwdProblem import *
from invProblem2d import *
from rectangle import *
from fista2 import *
from rto_scalable import *
import pickle
import os

np.random.seed(187762)

rect = Rectangle((0,0), (1,1), resol=5) # resol=5
rectCoarse = Rectangle((0,0), (1,1), resol=4)
gamma = 0.002
kappa = 1
N_obs = 49 # N_obs = 100


def myUTruth2(x, y):
	return np.logical_and(np.logical_and((x-1)**2 + y**2 <= 0.65**2, (x-1)**2 + y**2 > 0.55**2), x<0.85)*-1/log10(e) + np.cos(6*y)*np.sin(4*x)*0.2/log10(e) - 0.9/log10(e)*np.logical_and(np.logical_and( x< 0.4 , x >= 0.3375) ,y >= 0.725 - tol) 

def u_D_term(x, y):
	return np.logical_and(x <= 1e-10, True)*0.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] <= tol:
			return True
		else:
			return False

f = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.8)**2 + (y-.2)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
#m1 = Besov11Wavelet(rect, kappa, 1.0, 5) # resol = 5

s = 1.8

m1 = GeneralizedGaussianWavelet2d(rect, kappa, s, 5)
invProb = inverseProblem(fwd, m1, gamma)


#obspos = np.random.uniform(0, 1, (2, N_obs))
#obspos = [obspos[0,:], obspos[1, :]]
obsposx = np.linspace(0.05, 0.95, 7)
obsposy = np.linspace(0.05, 0.95, 7)
OX, OY = np.meshgrid(obsposx, obsposy)
obspos = np.stack((OX, OY)).reshape((2, 49))
obspos = [obspos[0,:], obspos[1, :]]
invProb.obspos = obspos

uTruthCoarse = mor.mapOnRectangle(rectCoarse, "handle", lambda x, y: myUTruth2(x, y))
uTruth = mor.mapOnRectangle(rect, "wavelet", uTruthCoarse.waveletcoeffs)
obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.obs = obs
invProb.plotSolAndLogPermeability(uTruth, obs=obs, blocky=True)



tau = 0.0001

x0 = unpackWavelet(m1._mean)
I_fnc = lambda x: invProb.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
Phi_fnc = lambda x:  invProb.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
DPhi_fnc = lambda x: np.array(invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)), version=0))
DNormpart_fnc = lambda x: m1.multiplyWithInvCov(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
cutoffmultiplier = sqrt(kappa) #2*gamma**2 * sqrt(kappa)
alpha0 = tau
N_iter = 1500 #100


filename="data_s" + str(s) + ".pickle"

# check whether there is data
if os.path.isfile(filename):
# load optimizer
	with open(filename, 'rb') as f:
		# The protocol version used is detected automatically, so we do not
		# have to specify it.
		result = pickle.load(f)
else:
	start = time.time()
	result = gradDesc_newBacktracking(x0, I_fnc, Phi_fnc, DPhi_fnc, DNormpart_fnc, alpha0=alpha0, eta=0.5, N_iter=N_iter, showDetails=True)#FISTA(x0, I_fnc, Phi_fnc, DPhi_fnc, cutoffmultiplier, alpha0=alpha0, eta=0.5, N_iter=N_iter, backtracking=True, c=1.0, showDetails=True)
	end = time.time()
	with open(filename, 'wb') as f:
		# Pickle the 'data' dictionary using the highest protocol available.
		pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

xk, Is, Phis, num_bt = result["xk"], result["Is"], result["Phis"], result["num_backtrackings"]
uOpt = mor.mapOnRectangle(rect, "wavelet", packWavelet(xk[-1]))



#uOpt2 = mor.mapOnRectangle(rect, "wavelet", packWavelet(result2.x*0.0000001))

plt.figure(); plt.ion();
plt.subplot(311)
plt.plot(Is, label = "I")
plt.legend()
plt.subplot(312)
plt.plot(Phis, label = "Phi")
plt.legend()
plt.subplot(313)
plt.plot(Is-Phis, label = "norm")
plt.legend()
invProb.plotSolAndLogPermeability(uOpt, obs=obs, blocky=True)

print("functional value I of ground truth: " + str(invProb.I(uTruth, obs)))
print("squared Cameron-Martin norm of ground truth: " + str(invProb.prior.normpart(uTruth)))
print("squared misfit of ground Truth: " + str(invProb.Phi(uTruth, obs)))


print("functional value I of backtracking-FISTA optimizer: " + str(invProb.I(uOpt, obs)))
print("squared Cameron-Martin norm of backtracking-FISTA optimizer: " + str(invProb.prior.normpart(uOpt)))
print("squared misfit of backtracking-FISTA optimizer: " + str(invProb.Phi(uOpt, obs)))

invProb.fcounter = 0;
invProb.Jfcounter = 0;

def f_fwd(theta):
	#print(invProb.fcounter);
	invProb.fcounter += 1
	return invProb.Gfnc(mor.mapOnRectangle(rect, "wavelet", packWavelet(theta)))
def Jf_fwd(theta):
	#print(invProb.Jfcounter);
	invProb.Jfcounter += 1
	return invProb.DG_adjoint_vec_wavelet(mor.mapOnRectangle(rect, "wavelet", packWavelet(theta)), version=0)
y = invProb.obs
sigma = gamma
Gamma = m1.Cov()
theta0 = unpackWavelet(uOpt.waveletcoeffs)
N_samples = 150

def optimizer(x0, I_fnc, DI_fnc):
	return gradDesc_general(x0, I_fnc, DI_fnc, alpha0=alpha0, eta=0.5, N_iter=200, showDetails=True)

res = rto_scalable(f_fwd, Jf_fwd, y, sigma, Gamma, theta0, mapEstimator = theta0, mean_theta=None, N_samples=N_samples, init_method="fixed", opt_with_grad=True)
#res = rto_l1(f, Jf, invProb.obs, gamma, kappa*np.ones(theta0.shape), theta0, N_samples = 30, init_method="random")
#res = rto(f, Jf2, invProb.obs, gamma, kappa, theta0, mean_theta=None, N_samples=3, init_method="MAP")
print("Finished sampling")
print(time.asctime(time.localtime(time.time())))
samples = res["samples_corrected"]
samples_plain = res["samples_plain"]
thetaMAP = res["thetaMAP"]
weights = res["weights"]
weights_old = res["weights_old"]
logweights = res["logweights"]

plt.figure(); plt.subplot(311)
plt.semilogy(weights)
plt.title("weights")
plt.subplot(312)
plt.plot(logweights)
plt.title("logweights")
plt.subplot(313)
plt.semilogy(weights_old)
plt.title("weights_old")

invProb.plotSolAndLogPermeability(mor.mapOnRectangle(rect, "wavelet", packWavelet(thetaMAP)), obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(mor.mapOnRectangle(rect, "wavelet", packWavelet(samples[0])), obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(mor.mapOnRectangle(rect, "wavelet", packWavelet(samples[1])), obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(mor.mapOnRectangle(rect, "wavelet", packWavelet(samples[2])), obs=obs, blocky=True)

Is = []
Phis = []
Is_plain = []
Phis_plain = []
for samp in samples:
	Is.append(invProb.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(samp))))
	Phis.append(invProb.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(samp))))
	
for samp in samples_plain:
	Is_plain.append(invProb.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(samp))))
	Phis_plain.append(invProb.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(samp))))

mean_wc = np.zeros_like(samples_plain[0])
for samp in samples_plain:
	mean_wc += samp

mean_wc /= len(samples_plain)
invProb.plotSolAndLogPermeability(mor.mapOnRectangle(rect, "wavelet", packWavelet(mean_wc)), obs=obs, blocky=True)


plt.figure()
plt.subplot(211)
plt.plot(Is)
plt.plot(Is_plain)
plt.subplot(212)
plt.plot(Phis)
plt.plot(Phis_plain)

Gvals = np.zeros((N_samples, len(obs)))
for n, samp in enumerate(samples):
	Gvals[n, :] = invProb.Gfnc(mor.mapOnRectangle(rect, "wavelet", packWavelet(samp)))

plt.figure()
for k in range(N_obs):
	plt.subplot(7,7,k+1)
	plt.plot(Gvals[:, k])
	plt.hlines(y[k], xmin=0, xmax=N_samples-1)	
	plt.hlines([y[k]-gamma, y[k], y[k]+gamma], colors=['r', 'k', 'r'], xmin=0, xmax=N_samples-1)	

print("Finished computation of Phis")
print(time.asctime(time.localtime(time.time())))

"""f0 = f(theta0)
Jf0 = Jf(theta0)
direction = np.random.normal(0, 0.01, theta0.shape)
factors = np.linspace(0, 1, 20)
fs = np.zeros((len(f0),len(factors)))
fs_lin = np.zeros((len(f0),len(factors)))
for m, fac in enumerate(factors):
	fs[:, m] = (f(theta0+fac*direction))
	fs_lin[:, m] = f(theta0) + np.dot(Jf0, fac*direction)

plt.figure()
plt.plot(fs[17, :])
plt.plot(fs_lin[17, :])

plt.figure();
plt.title("with correction")
plt.plot(Guph-Gu, 'gray')
plt.plot(GuTaylor-Gu, 'red')
plt.figure();
plt.title("without correction")
plt.plot(Guph-Gu, 'gray')
plt.plot(GuTaylor_-Gu, 'red')


uList, uListUnique, PhiList = invProb.randomwalk_pCN(uOpt, 6500000, beta=0.025, showDetails=True) #7,000,000
plt.figure(); plt.plot(PhiList)

invProb.plotSolAndLogPermeability(uList[500000], obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(uList[2000000], obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(uList[3500000], obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(uList[5000000], obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(uList[6300000], obs=obs, blocky=True)

data = {'uList': uList, 'uListUnique': uListUnique, 'PhiList': PhiList}
with open('data.pickle', 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

f = lambda wc_unp: invProb.Gfnc(mor.mapOnRectangle(rect, "wavelet", packWavelet(wc_unp)))


#rto_l1(f, Jf, y, sigma, lambdas, u0, N_samples = 10, init_method="random")

takeeveryNth = 25
uwcs_mean = 0;
uwcs_var = 0;
counter = 0;

u_var = 0;

for n in range(0, 6500000, takeeveryNth):
	uwcs_mean += unpackWavelet(uList[n].waveletcoeffs)
	counter += 1

uwcs_mean = uwcs_mean/counter

#for n in range(0, 6500000, takeeveryNth):
#	uwcs_var += (unpackWavelet(uList[n])-uwcs_mean)**2

#uwcs_var = uwcs_var/(counter-1)


#uwcs_std = np.sqrt(uwcs_var)


uMean = mor.mapOnRectangle(rect, "wavelet", packWavelet(uwcs_mean))
invProb.plotSolAndLogPermeability(uMean, obs=obs, blocky=True)
#uStd = mor.mapOnRectangle(rect, "wavelet", packWavelet(uwcs_std))
#plt.figure(); plt.imshow(np.flipud((uStd.values)**2), extent=[0, 1, 0, 1], cmap=plt.cm.viridis, interpolation='none')
#plt.colorbar()



for n in range(0, 6500000, takeeveryNth):
	u_var += (uList[n].values-uMean.values)**2

u_var = u_var/(counter-1)
u_var_fnc = mor.mapOnRectangle(rect, "expl", (u_var))

u_std = mor.mapOnRectangle(rect, "expl", np.sqrt(u_var))

plt.figure(); plt.imshow(np.flipud((u_std.values)), extent=[0, 1, 0, 1], cmap=plt.cm.viridis, interpolation='none')
plt.colorbar()

norms = np.zeros((6500000,))

for m in range(6500000):
	norms[m] = invProb.prior.normpart(uList[m])
	
plt.figure();
plt.subplot(311)
plt.plot(PhiList[1:]+norms, label="I(sample)")
plt.legend(loc=4)
plt.subplot(312)
plt.plot(PhiList, label="Phi(sample)")
plt.legend(loc=4)
plt.subplot(313)
plt.plot(norms, label="norm(sample)")
plt.legend(loc=4)

# plot mean+1std
invProb.plotSolAndLogPermeability(uMean+u_std, obs=obs, blocky=True)
# plot mean-1std
invProb.plotSolAndLogPermeability(uMean+(u_std*(-1)), obs=obs, blocky=True)

plt.figure(); plt.imshow(np.flipud((u_std.values)), extent=[0, 1, 0, 1], cmap=plt.cm.viridis, interpolation='none', vmin=-3, vmax = 1)"""

