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

rect = Rectangle((0,0), (1,1), resol=7)
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

s = 1.5

m1 = GeneralizedGaussianWavelet2d(rect, 1.0, s, 6)
m2 = GaussianFourier2d(rect, np.zeros((33,33)), 0.5, 2.0)
invProb = inverseProblem(fwd, m1, gamma)
invProb2 = inverseProblem(fwd, m2, gamma)
uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: uTruth_handle(x, y))
	


N_obs = 49
N_iter = 10
obspos = np.random.uniform(0, 1, (2, N_obs))
obspos = [obspos[0,:], obspos[1, :]]
obsposx = np.linspace(0.05, 0.95, 7)
obsposy = np.linspace(0.05, 0.95, 7)
OX, OY = np.meshgrid(obsposx, obsposy)
obspos = np.stack((OX, OY)).reshape((2, 49))
obspos = [obspos[0,:], obspos[1, :]]
invProb.obspos = obspos
invProb2.obspos = obspos


obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.obs = obs
invProb2.obs = obs
invProb.plotSolAndLogPermeability(uTruth, obs=obs, blocky=True)
invProb2.plotSolAndLogPermeability(uTruth, obs=obs, blocky=True)

u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)

tau = 0.000005




x0 = unpackWavelet(m1._mean)
x02 = np.zeros((33,33))
alpha0 = tau
I_fnc_wavelet_quick = lambda x: invProb.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
I_fnc_fourier_quick = lambda x: invProb.I(mor.mapOnRectangle(rect, "fourier", x.reshape((33,33))))
Phi_fnc_wavelet_quick = lambda x:  invProb.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
Phi_fnc_fourier_quick = lambda x:  invProb.Phi(mor.mapOnRectangle(rect, "fourier", x.reshape((33,33))))
DPhi_fnc_wavelet_quick = lambda x: np.array(invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)), version=0))
DPhi_fnc_fourier_quick = lambda x: invProb2.DPhi_adjoint_vec_fourier(mor.mapOnRectangle(rect, "fourier", x.reshape((33,33))), version=0).reshape((-1,))
DPhi_fnc_wavelet_slow = lambda x: np.array(invProb.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)), version=2))
DPhi_fnc_fourier_slow = lambda x: invProb2.DPhi_adjoint_vec_fourier(mor.mapOnRectangle(rect, "fourier", x.reshape((33,33))), version=2).reshape((-1,))
DNormpart_fnc_wavelet_quick = lambda x: invProb.DNormpart(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
DNormpart_fnc_fourier_quick = lambda x: m2.covProd(mor.mapOnRectangle(rect, "fourier", x.reshape((33,33)))).reshape((-1,))
DI_fnc = lambda x: DPhi_fnc_wavelet_quick(x) + DNormpart_fnc_wavelet_quick(x)

u02 = mor.mapOnRectangle(rect, "fourier", x02)

"""
start=time.time()
DPhi_fnc_wavelet_quick(x0)
print("1")
e1=time.time()
DPhi_fnc_wavelet_slow(x0)
print("2")
e2=time.time()
DPhi_fnc_fourier_quick(x02.reshape((-1,)))
print("3")
e3=time.time()
DPhi_fnc_fourier_slow(x02.reshape((-1,)))
print("4")
e4=time.time()

print(e1-start)
print(e2-e1)
print(e3-e2)
print(e4-e3)"""


N_iter1 = 4000
N_iter2 = 150
N_iter3 = 90
N_iter4 = 40
result_wavelet_quick = gradDesc_newBacktracking(x0, I_fnc_wavelet_quick, Phi_fnc_wavelet_quick, DPhi_fnc_wavelet_quick, DNormpart_fnc_wavelet_quick, alpha0=alpha0, eta=0.5, N_iter=N_iter1, showDetails=True)
result_fourier_quick = gradDesc_newBacktracking(x02.reshape((-1,)), I_fnc_fourier_quick, Phi_fnc_fourier_quick, DPhi_fnc_fourier_quick, DNormpart_fnc_fourier_quick, alpha0=alpha0, eta=0.5, N_iter=N_iter2, showDetails=True)
result_wavelet_slow = gradDesc_newBacktracking(x0, I_fnc_wavelet_quick, Phi_fnc_wavelet_quick, DPhi_fnc_wavelet_slow, DNormpart_fnc_wavelet_quick, alpha0=alpha0, eta=0.5, N_iter=N_iter3, showDetails=True)
result_fourier_slow = gradDesc_newBacktracking(x02.reshape((-1,)), I_fnc_fourier_quick, Phi_fnc_fourier_quick, DPhi_fnc_fourier_slow, DNormpart_fnc_fourier_quick, alpha0=alpha0, eta=0.5, N_iter=N_iter4, showDetails=True)

xk, Is, Phis, num_bt, times = result_wavelet_quick["xk"], result_wavelet_quick["Is"], result_wavelet_quick["Phis"], result_wavelet_quick["num_backtrackings"], result_wavelet_quick["times"]
xk2, Is2, Phis2, num_bt2, times2 = result_fourier_quick["xk"], result_fourier_quick["Is"], result_fourier_quick["Phis"], result_fourier_quick["num_backtrackings"], result_fourier_quick["times"]
xk3, Is3, Phis3, num_bt3, times3 = result_wavelet_slow["xk"], result_wavelet_slow["Is"], result_wavelet_slow["Phis"], result_wavelet_slow["num_backtrackings"], result_wavelet_slow["times"]
xk4, Is4, Phis4, num_bt4, times4 = result_fourier_slow["xk"], result_fourier_slow["Is"], result_fourier_slow["Phis"], result_fourier_slow["num_backtrackings"], result_fourier_slow["times"]
uOpt = mor.mapOnRectangle(rect, "wavelet", packWavelet(xk[-1]))
uOpt2 = mor.mapOnRectangle(rect, "fourier", xk2[-1].reshape((33,33)))
uOpt3 = mor.mapOnRectangle(rect, "wavelet", packWavelet(xk3[-1]))
uOpt4 = mor.mapOnRectangle(rect, "fourier", xk4[-1].reshape((33,33)))
invProb.plotSolAndLogPermeability(uOpt, obs=obs, blocky=True)
invProb2.plotSolAndLogPermeability(uOpt2, obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(uOpt3, obs=obs, blocky=True)
invProb2.plotSolAndLogPermeability(uOpt4, obs=obs, blocky=True)
invProb.plotSolAndLogPermeability(uOpt, obs=obs, blocky=False)
invProb2.plotSolAndLogPermeability(uOpt2, obs=obs, blocky=False)
invProb.plotSolAndLogPermeability(uOpt3, obs=obs, blocky=False)
invProb2.plotSolAndLogPermeability(uOpt4, obs=obs, blocky=False)
times_normalized = [t-times[0] for t in times]
times_normalized2 = [t-times2[0] for t in times2]
times_normalized3 = [t-times3[0] for t in times3]
times_normalized4 = [t-times4[0] for t in times4]

plt.figure(); plt.ion();
plt.subplot(311)
plt.title("quick wavelet")
plt.plot(times_normalized, Is, '-', label = "I")
plt.legend()
plt.subplot(312)
plt.plot(times_normalized, Phis, '-', label = "Phi")
plt.legend()
plt.subplot(313)
plt.plot(times_normalized, Is-Phis, '-', label = "norm")
plt.legend()


plt.figure(); plt.ion();
plt.subplot(311)
plt.title("quick fourier")
plt.plot(times_normalized2, Is2, '-', label = "I")
plt.legend()
plt.subplot(312)
plt.plot(times_normalized2, Phis2, '-', label = "Phi")
plt.legend()
plt.subplot(313)
plt.plot(times_normalized2, Is2-Phis2, '-', label = "norm")
plt.legend()

plt.figure(); plt.ion();
plt.title("I (fourier mth2)")
plt.plot(times_normalized2, Is2, '-', label = "I")


plt.figure(); plt.ion();
plt.subplot(311)
plt.title("slow wavelet")
plt.plot(times_normalized3, Is3, '-', label = "I")
plt.legend()
plt.subplot(312)
plt.plot(times_normalized3, Phis3, '-', label = "Phi")
plt.legend()
plt.subplot(313)
plt.plot(times_normalized3, Is3-Phis3, '-', label = "norm")
plt.legend()


plt.figure(); plt.ion();
plt.subplot(311)
plt.title("slow fourier")
plt.plot(times_normalized4, Is4, '-', label = "I")
plt.legend()
plt.subplot(312)
plt.plot(times_normalized4, Phis4, '-', label = "Phi")
plt.legend()
plt.subplot(313)
plt.plot(times_normalized4, Is4-Phis4, '-', label = "norm")
plt.legend()

"""
plt.figure(); plt.ion();
plt.subplot(311)
plt.title("all")
plt.plot(times_normalized, Is, '.-', label = "I qu wa")
plt.plot(times_normalized2, Is2, '.-', label = "I qu fo")
plt.plot(times_normalized3, Is3, '.-', label = "I sl wa")
plt.plot(times_normalized4, Is4, '.-', label = "I sl fo")
plt.legend()
plt.subplot(312)
plt.plot(times_normalized, Phis, '.-', label = "Phi qu wa")
plt.plot(times_normalized2, Phis2, '.-', label = "Phi qu fo")
plt.plot(times_normalized3, Phis3, '.-', label = "Phi sl wa")
plt.plot(times_normalized4, Phis4, '.-', label = "Phi sl fo")
plt.legend()
plt.subplot(313)
plt.plot(times_normalized, Is-Phis, '.-', label = "norm qu wa")
plt.plot(times_normalized2, Is2-Phis2, '.-', label = "norm qu fo")
plt.plot(times_normalized3, Is3-Phis3, '.-', label = "norm sl wa")
plt.plot(times_normalized4, Is4-Phis4, '.-', label = "norm sl fo")
plt.legend()
"""

plt.figure(); plt.ion();
plt.subplot(311)
plt.title("all")
plt.semilogy(times_normalized, Is, '.-', label = "I qu wa")
plt.semilogy(times_normalized2, Is2, '.-', label = "I qu fo")
plt.semilogy(times_normalized3, Is3, '.-', label = "I sl wa")
plt.semilogy(times_normalized4, Is4, '.-', label = "I sl fo")
plt.legend()
plt.subplot(312)
plt.semilogy(times_normalized, Phis, '.-', label = "Phi qu wa")
plt.semilogy(times_normalized2, Phis2, '.-', label = "Phi qu fo")
plt.semilogy(times_normalized3, Phis3, '.-', label = "Phi sl wa")
plt.semilogy(times_normalized4, Phis4, '.-', label = "Phi sl fo")
plt.legend()
plt.subplot(313)
plt.plot(times_normalized, Is-Phis, '.-', label = "norm qu wa")
plt.plot(times_normalized2, Is2-Phis2, '.-', label = "norm qu fo")
plt.plot(times_normalized3, Is3-Phis3, '.-', label = "norm sl wa")
plt.plot(times_normalized4, Is4-Phis4, '.-', label = "norm sl fo")
plt.legend()



plt.figure(); plt.ion();
plt.semilogy(times_normalized, Is, 'b-', label = "I (wavelet, mth2)")
plt.semilogy(times_normalized3, Is3, 'b--', label = "I (wavelet, mth1)")
plt.semilogy(times_normalized4, Is4, 'g-', label = "I (fourier, mth1)")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("I(iteration)")

plt.figure(); plt.ion();
plt.semilogy(times_normalized, Is-Phis, 'b-', label = "I (wavelet, mth2)")
plt.semilogy(times_normalized3, Is3-Phis3, 'b--', label = "I (wavelet, mth1)")
plt.semilogy(times_normalized4, Is4-Phis4, 'g-', label = "I (fourier, mth1)")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("norm(iteration)")

plt.figure(); plt.ion();
plt.semilogy(Is, 'b-', label = "I (wavelet, mth2)")
plt.semilogy(Is3, 'b--', label = "I (wavelet, mth1)")
plt.semilogy(Is4, 'g-', label = "I (fourier, mth 1)")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("I(iteration)")

plt.figure(); plt.ion();
plt.subplot(311)
plt.title("all")
plt.semilogy(times_normalized, Is, '-', label = "I qu wa")
plt.semilogy(times_normalized3, Is3, '-', label = "I sl wa")
plt.semilogy(times_normalized4, Is4, '-', label = "I sl fo")
plt.legend()
plt.subplot(312)
plt.semilogy(times_normalized, Phis, '-', label = "Phi qu wa")
plt.semilogy(times_normalized3, Phis3, '-', label = "Phi sl wa")
plt.semilogy(times_normalized4, Phis4, '-', label = "Phi sl fo")
plt.legend()
plt.subplot(313)
plt.plot(times_normalized, Is-Phis, '-', label = "norm qu wa")
plt.plot(times_normalized3, Is3-Phis3, '-', label = "norm sl wa")
plt.plot(times_normalized4, Is4-Phis4, '-', label = "norm sl fo")
plt.legend()

"""kappa = 10
m2 = Besov11Wavelet(rect, kappa, 1.0, 5) # resol = 5
invProb2 = inverseProblem(fwd, m2, gamma)


invProb2.obspos = obspos
invProb2.obs = obs


x0 = unpackWavelet(m1._mean)
I_fnc_wavelet_quick = lambda x: invProb2.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
Phi_fnc_wavelet_quick = lambda x:  invProb2.Phi(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)))
DPhi_fnc_wavelet_quick = lambda x: np.array(invProb2.DPhi_adjoint_vec_wavelet(mor.mapOnRectangle(rect, "wavelet", packWavelet(x)), version=0))
cutoffmultiplier = 2*gamma**2 * sqrt(kappa) #sqrt(kappa)
alpha0 = tau
N_iter = 100 #100

start = time.time()
result_wavelet_quick = FISTA(x0, I_fnc_wavelet_quick, Phi_fnc_wavelet_quick, DPhi_fnc_wavelet_quick, cutoffmultiplier, alpha0=alpha0, eta=0.5, N_iter=N_iter, backtracking=True, c=1.0, showDetails=True)
end = time.time()

xk, Is, Phis, num_bt = result_wavelet_quick["xk"], result_wavelet_quick["Is"], result_wavelet_quick["Phis"], result_wavelet_quick["num_backtrackings"]
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
ustar = mor.mapOnRectangle(rect, "fourier", uOpt2.fouriermodes)
DPhiu = invProb2.DPhi_adjoint_vec_fourier(ustar, version=0)
DPhiu_2 = invProb.DPhi_adjoint_vec_fourier(ustar, version=2)

#direction = mor.mapOnRectangle(rect, "wavelet", packWavelet(np.random.normal(0, 0.0005, unpackWavelet(ustar.waveletcoeffs).shape)))
direction = mor.mapOnRectangle(rect, "fourier", np.random.normal(0, 0.05, (33,33)))
rs = np.linspace(-1, 1, 15)
Phiu = invProb.Phi(ustar)
Phis = np.zeros(rs.shape)
PhisLin = np.zeros(rs.shape)
PhisLin_2 = np.zeros(rs.shape)
for n, r in enumerate(rs):
	Phis[n] = invProb.Phi(ustar + direction*r)
	#PhisLin[n] = Phiu + np.dot(DPhiu, unpackWavelet(direction.waveletcoeffs)*r)
	PhisLin[n] = Phiu + np.dot(DPhiu, direction.fouriermodes.reshape((-1,))*r)
	#PhisLin_2[n] = Phiu + np.dot(DPhiu_2, unpackWavelet(direction.waveletcoeffs)*r)
	PhisLin_2[n] = Phiu + np.dot(DPhiu_2, direction.fouriermodes.reshape((-1,))*r)

plt.figure();
plt.plot(rs, Phis, 'b.-')
plt.plot(rs, PhisLin, 'r.-')
plt.plot(rs, PhisLin_2, 'g.-')"""


