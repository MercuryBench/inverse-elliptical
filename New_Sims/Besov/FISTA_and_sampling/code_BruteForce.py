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
import pickle

np.random.seed(1877262)

rect = Rectangle((0,0), (1,1), resol=5)
gamma = 0.01
kappa = (500)**2
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



tau = 0.0001

x0 = unpackWavelet(m1._mean)
I_fnc = lambda x: invProb.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
Phi_fnc = lambda x:  invProb.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
DPhi_fnc = lambda x: np.array(invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(rect, "wavelet", packWavelet(x))))
cutoffmultiplier = sqrt(kappa) #2*gamma**2 * sqrt(kappa)
alpha0 = tau
N_iter = 200 #100

start = time.time()
result = FISTA(x0, I_fnc, Phi_fnc, DPhi_fnc, cutoffmultiplier, alpha0=alpha0, eta=0.5, N_iter=N_iter, backtracking=True, c=1.0, showDetails=True)
end = time.time()

xk, Is, Phis, num_bt = result["xk"], result["Is"], result["Phis"], result["num_backtrackings"]
uOpt = mor.mapOnRectangle(rect, "wavelet", packWavelet(xk[-1]))

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

plt.figure(); plt.imshow(np.flipud((u_std.values)), extent=[0, 1, 0, 1], cmap=plt.cm.viridis, interpolation='none', vmin=-3, vmax = 1)

