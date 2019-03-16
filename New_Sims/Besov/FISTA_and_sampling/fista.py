from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
import math
from math import e
import time

# assume I = Phi + kappa*|.|_1

def shrinkage(z, cutoff):
	retvals = np.zeros((len(z),))
	#cutoff = 2*tau*gamma**2*sqrt(kappa)*5000
	#cutoff = 2*steptau*gamma**2*kappa
	for k in range(len(z)):
		if z[k] >= cutoff:
			retvals[k] = z[k]-cutoff
		elif z[k] >= -cutoff:
			retvals[k] = 0
		else: 
			retvals[k] = z[k]+cutoff
	return retvals

def FISTA(x0, I_fnc, Phi_fnc, DPhi_fnc, cutoffmultiplier, alpha0=1.0, eta=0.5, N_iter=500, backtracking=True, c=1.0, showDetails=False):
	start = time.time()
	xk = np.zeros((N_iter, x0.size))
	xk[0, :] = x0
	yk = np.zeros((N_iter, x0.size))
	yk[1, :] = x0
	tk = np.zeros((N_iter,))
	tk[1] = 1
	Is = np.zeros((N_iter,))
	Is[0] = I_fnc(x0)
	Phis = np.zeros((N_iter,))
	Phis[0] = Phi_fnc(x0)
	
	alpha = alpha0
	
	# for running diagnostics:
	num_backtrackings = np.zeros((N_iter,))
	for k in range(N_iter):
		DPhi = DPhi_fnc(yk[k, :])
		if backtracking:
			alpha = alpha0
			proposal = shrinkage(yk[k, :] - alpha*DPhi, cutoffmultiplier*alpha)
			max_backtrack = 20;
			#while I_fnc(proposal) > Phi_fnc(yk[k, :]) + np.dot(DPhi.T, proposal-yk[k, :]) + 1.0/(2.0*alpha*c)*np.dot((proposal-yk[k, :]).T, proposal-yk[k, :]) + I_fnc(proposal) - Phi_fnc(proposal): ## ALTERNATIVE
			while np.isnan(Phi_fnc(proposal)) or Phi_fnc(proposal) > Phi_fnc(yk[k, :]) + np.dot(DPhi.T, proposal-yk[k, :]) + 1.0/(2.0*alpha*c)*np.dot((proposal-yk[k, :]).T, proposal-yk[k, :]): ## ALTERNATIVE
				alpha = alpha*eta
				proposal = shrinkage(yk[k, :] - alpha*DPhi,  cutoffmultiplier*alpha)
				max_backtrack -= 1
				num_backtrackings[k] += 1
				if max_backtrack <= 0:
					break
			xk[k, :] = proposal
		else:
			xk[k, :] = shrinkage(yk[k, :] - alpha*DPhi,  cutoffmultiplier*alpha)
		Is[k] = I_fnc(xk[k, :])
		Phis[k] = Phi_fnc(xk[k, :])
		if k < N_iter-1: # preparation for next step only needed up to penultimate iteration
			tk[k+1] = 0.5 * (1.0 + sqrt(1.0 + 4.0*tk[k]**2))
			yk[k+1, :] = xk[k, :] + (tk[k]-1)/tk[k+1] * (xk[k, :] - xk[k-1, :])
	result = {"xk": xk, "Is": Is, "Phis": Phis, "num_backtrackings": num_backtrackings}
	end = time.time()
	if showDetails:
		print("Took " + str(end-start) + " seconds")
		print("Reduction of function value from " + str(Is[0]) + " to " + str(Is[-1]))
		print("Function value consists of")
		print("Phi(u)  = " + str(Phis[-1]))
		print("norm(u) = " + str(Is[-1] - Phis[-1]))
	return result

	
if __name__ == "__main__":
	N = 100
	np.random.seed(1992)
	xs_obs = np.random.uniform(0, 3, (N,))

	N_modes = 19
	kappa = 5.0
	gamma = 0.5


	def f(u, xs):
		temp = u[0]
		for k in range(N_modes):
			temp += u[k+1] * 1/(k+1)*np.cos((k+1)*xs)
		for k in range(N_modes):
			temp += u[k+N_modes+1] * 1/(k+1)* np.sin((k+1)*xs)
		return temp

	A = np.zeros((N, 2*N_modes+1))
	A[:, 0] = np.ones((N,))
	for k in range(N_modes):
		A[:, k+1] =  1/(k+1)*np.cos((k+1)*xs_obs)
	for k in range(N_modes):
		A[:, k+N_modes+1] =  1/(k+1)*np.sin((k+1)*xs_obs)

	def norm1(u):
		return kappa*np.sum(np.abs(u))
	def Misfit(u, y):
		misfit = y - np.dot(A, u)
		return 1.0/(2.0*gamma**2)*np.dot(misfit.T, misfit) 
	def FncL1(u, y):
		return Misfit(u, y) + norm1(u)

	DMisfit = lambda u, y: gamma**(-2)*np.dot(A.T, np.dot(A, u)-y)

	uTruth = np.array([-1.0, 1.0, -1.3, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 2.2, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	x = np.arange(0, 3, 0.01)
	y = f(uTruth, x)

	obs = f(uTruth, xs_obs) + np.random.normal(0, gamma, (len(xs_obs),))

	plt.figure(1);plt.ion()
	plt.plot(x, y, 'g')
	plt.plot(xs_obs, obs, 'rx')
	plt.show()

	xk, Is, Phis = FISTA(np.zeros((uTruth.size,)), lambda x: FncL1(x, obs), lambda x: Misfit(x, obs), lambda x: DMisfit(x, obs), 2*gamma**2*kappa, alpha0=0.0001, eta=0.5, N_iter=500, backtracking=False)
	xk_BT, Is_BT, Phis_BT = FISTA(np.zeros((uTruth.size,)), lambda x: FncL1(x, obs), lambda x: Misfit(x, obs), lambda x: DMisfit(x, obs), 2*gamma**2*kappa, alpha0=0.01, eta=0.5, N_iter=500, backtracking=True)
	plt.plot(x, f(xk[-1, :], x), 'b', label="without backtracking")
	plt.plot(x, f(xk_BT[-1, :], x), 'k', label="with backtracking")
	plt.legend()

	plt.figure();
	plt.subplot(311)
	plt.plot(Is, 'b-', label="Is")
	plt.plot(Is_BT, 'k-', label="Is_BT")
	plt.legend()
	plt.subplot(312)
	plt.plot(Phis, 'b-', label="Phis")
	plt.plot(Phis_BT, 'k-', label="Phis_BT")
	plt.legend()
	plt.subplot(313)
	plt.plot(Is-Phis, 'b-', label="norms")
	plt.plot(Is_BT-Phis_BT, 'k-', label="norms_BT")
	plt.legend()


	print("ground truth: I = " + str(FncL1(uTruth, obs)) + " = " + str(Misfit(uTruth, obs)) + " (Phi) + " + str(norm1(uTruth)) + " (norm)")
	print("optimizer: I = " + str(FncL1(xk[-1, :], obs)) + " = " + str(Misfit(xk[-1, :], obs)) + " (Phi) + " + str(norm1(xk[-1, :])) + " (norm)")
	print("optimizer (with BT): I = " + str(FncL1(xk_BT[-1, :], obs)) + " = " + str(Misfit(xk_BT[-1, :], obs)) + " (Phi) + " + str(norm1(xk_BT[-1, :])) + " (norm)")

	plt.figure();
	plt.plot(uTruth, '.-g')
	plt.plot(xk[-1, :], '.-r')
	plt.plot(xk_BT[-1, :], '.-k')
	

	
