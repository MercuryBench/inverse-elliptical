from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi
from fwdProblem import *
from measures import *
import mapOnInterval as moi

class inverseProblem():
	def __init__(self, fwd, prior, gamma, obsind=None, obs=None):
		# need: type(fwd) == fwdProblem, type(prior) == measure
		self.fwd = fwd
		self.prior = prior
		self.obsind = obsind
		self.obs = obs
		self.gamma = gamma
		
	# Forward operators and their derivatives:	
	def Ffnc(self, x, u, g=None): # F is like forward, but uses logpermeability instead of permeability
		# if g == None, then we take the default right hand side
		perm = moi.mapOnInterval("handle", lambda x: np.exp(u.handle(x)))
		return self.fwd(x, perm)
	def DFfnc(self, x, u, h):
		p = F(x, u)
		pprime = moi.differentiate(x, p)
		tempfnc = moi.mapOnInterval("expl", h.values*np.exp(u.values)*pprime
		rhs = moi.differentiate(x, tempfnc)
		p1 = self.Ffnc(x, u, rhs, 0, 0)
"""		
def DF_long(x, u, g, pplus, pminus, h): # RECOMMENDED
	p, C = F(x, u, g, pplus, pminus)
	pprime = differentiate(x, p)
	rhs = differentiate(x, h*np.exp(u)*pprime)
	p1, C = F(x, u, rhs, 0, 0)
	return p1

def D2F_long(x, u, g, pplus, pminus, h1, h2): # THIS WORKS! make this better: replace differentiation of trigonometric sums by algebraic method differentiate_modes. THIS NEEDS TO BE CHANGED TO THE STRANGE NEW FOURIER BASIS
	p, C = F(x, u, g, pplus, pminus)
	pprime = differentiate(x, p)
	rhs1 = differentiate(x, np.exp(u)*h1*pprime)
	rhs2 = differentiate(x, np.exp(u)*h2*pprime)
	p1, C = F(x, u, rhs1, 0.0, 0.0)
	p2, C = F(x, u, rhs2, 0.0, 0.0)
	p1prime = differentiate(x, p1)
	p2prime = differentiate(x, p2)
	rhs11 = differentiate(x, np.exp(u)*(h1*p2prime + h2*p1prime))
	rhs22 = differentiate(x, np.exp(u)*h1*h2*pprime)
	p22, C = F(x, u, rhs22, 0.0, 0.0)
	p11, C = F(x, u, rhs11, 0.0, 0.0)
	return p11+p22"""
	
	
	def Gfnc(self, x, u):
		if self.obsind == None:
			raise ValueError("obsind need to be defined")
		p = self.Ffnc(x, u)
		obs = p.handle(x)[self.obsind]
		return obs
	def DGfnc(self, x, u, h):
		pass
"""		
def DG(x, u, g, x0_ind, pplus, pminus, h):
	Dp = DF_long(x, u, g, pplus, pminus, h)
	return Dp[x0_ind]

def D2G(x, u, g, x0_ind, pplus, pminus, h1, h2):
	D2p = D2F_long(x, u, g, pplus, pminus, h1, h2)
	return D2p[x0_ind]"""
	
	
	def Phi(self, x, u, obs):
		discrepancy = obs-Gfnc(x, u)
		return 1/(2*self.gamma**2)*np.dot(discrepancy,discrepancy) 
	def I(self, x, u, obs):
		return self.Phi(x, u, obs) + 1.0/2*prior.covInnerProd(u, u)

if __name__ == "__main__":
	x = np.linspace(0, 1, 512)
	gamma = 0.1
	delta = 0.05
	
	# boundary values for forward problem
	# -(k * p')' = g
	# p(0) = pminus
	# p(1) = pplus
	pplus = 2.0
	pminus = 1.0	
	# right hand side of forward problem
	g = moi.mapOnInterval("handle", lambda x: 1.0*x*(1-x))	
	# construct forward problem
	fwd = linEllipt(g, pplus, pminus)
	
	# prior measure:
	alpha = 0.7
	beta = 0.5
	mean = np.zeros((31,))
	prior = GaussianFourier(mean, alpha, beta)
	
	u0 = prior.sample()
	k0 = moi.mapOnInterval("handle", lambda x: np.exp(u0.handle(x)))
	plt.figure(1)
	plt.ion()
	plt.plot(x, u0.handle(x))
	plt.show()
	
	# construct solution and observation
	p0 = fwd.solve(x, k0)
	x0_ind = range(50, 450, 50) # observation indices
	obs = p0.handle(x)[x0_ind] + np.random.normal(0, gamma, (len(x0_ind),))
	plt.figure(2)
	plt.plot(x, p0.handle(x))
	plt.plot(x[x0_ind], obs, 'r.')
	
	ip = inverseProblem(fwd, prior, gamma, x0_ind, obs)
	
"""
	# shortcut for I
	Ifnc = lambda u, u_modes: I(x, u, u_modes, g, x0_ind, pplus, pminus, y, beta, alpha, gamma)
	# shortcut for DI
	DI_vecfnc = lambda u, u_modes: DI_vec(x, u, u_modes, g, x0_ind, pplus, pminus, y, beta, alpha, gamma, len(u_modes))

	D2I_matrix = lambda u, u_modes: D2I_mat(x, u, u_modes, g, x0_ind, pplus, pminus, y, beta, alpha, gamma, len(u_modes))
	#D2I_matrix_savetime = lambda u, u_modes: D2I_mat_savetime(x, u, u_modes, x0_ind, y, beta, alpha, gamma, len(u_modes))

	Ifnc_var = lambda u_modes: Ifnc(evalmodes(u_modes, x), u_modes)
	DI_vecfnc_var = lambda u_modes: DI_vecfnc(evalmodes(u_modes, x), u_modes)
	D2I_matrix_var = lambda u_modes: D2I_matrix(evalmodes(u_modes, x), u_modes)
	Lhd = lambda u_modes: Likelihood(x, evalmodes(u_modes, x), u_modes, g, x0_ind, pplus, pminus, y, beta, alpha, gamma)
	
	# solve forward problem for log permeability u
	p, C = F(x, u, g, pplus, pminus)
	
	uHist_modes = randomwalk(u0, u0_modes, samplefncfromprior, PhiOfU, delta, 150, NBurnIn = 10)
	pHist = np.zeros((uHist_modes.shape[0], u0.shape[0]))
	IVal = np.zeros(len(uHist_modes))
	for n, uu_modes in enumerate(uHist_modes):
		uu = evalmodes(uu_modes, x)
		pHist[n, :], C = F(x, uu, g, pplus, pminus)
		IVal[n] = Ifnc_var(uu_modes)
	
	uu_modes_mean = np.mean(uHist_modes, axis=0)
	uu_mean = evalmodes(uu_modes_mean, x)
	p_mean, C = F(x, uu_mean, g, pplus, pminus)
	
	
	
	#uMAP, uMAP_modes, recI, recU = gradDescRaphson(Ifnc, DI_vecfnc, D2I_matrix, uu_mean, uu_modes_mean, 3, 2.0*delta)
	import scipy.optimize
	#res = scipy.optimize.minimize(Ifnc_var, uu_modes_mean, method='BFGS', jac=DI_vecfnc_var, options={'disp': True, 'maxiter': 10})
	res = scipy.optimize.minimize(Ifnc_var, uu_modes_mean, method='Newton-CG', jac = DI_vecfnc_var, hess=D2I_matrix_var, options={'disp': True, 'maxiter': 8})
	# compute the Hessian of the cost functional in the map point
	uMAP_modes = res.x
	uMAP = evalmodes(uMAP_modes, x)
	pMAP, C = F(x, uMAP, g, pplus, pminus)

	Hess = D2I_mat(x, uMAP, uMAP_modes, g, x0_ind, pplus, pminus, y, beta, alpha, gamma, len(uMAP_modes))
	evals, evecs = np.linalg.eig(Hess)
	if np.all(np.linalg.eigvals(Hess) > 0):
		print("Hessian is positive definite")
	else:
		print("Hessian is not positive definite")
	
	# Test gradients of F
	h = u - uMAP
	h_modes = u_modes - uMAP_modes 
	Dfuh = DF_long(x, uMAP, g, pplus, pminus, h)
	D2fuh = D2F_long(x, uMAP, g, pplus, pminus, h, h)

	h2 = 0.05*h
	h_modes2 = 0.05*h_modes
	Dfuh2 = DF_long(x, uMAP, g, pplus, pminus, h2)
	D2fuh2 = D2F_long(x, uMAP, g, pplus, pminus, h2, h2)

	
	# Test gradients of I, see whether quadratic approx is alright
	DIuh = DI(x, uMAP, uMAP_modes, g, x0_ind, pplus, pminus, y, beta, alpha, gamma, h, h_modes)
	D2Iuh = D2I(x, uMAP, uMAP_modes, g, x0_ind, pplus, pminus, y, beta, alpha, gamma, h, h, h_modes, h_modes)
	
	print("value of I in u: ", Ifnc_var(u_modes))
	print("value of I in uMAP: " , Ifnc_var(uMAP_modes))
	print("linear approx value of I in u: ", Ifnc_var(uMAP_modes) + DIuh)
	print("quadrt approx value of I in u: ", Ifnc_var(uMAP_modes) + DIuh + 0.5*D2Iuh)
	
	data = [x, mean, x0_ind, gamma, alpha, beta, delta, pplus, pminus, g, factor, u_modes, u, k, p, y, u0_modes, u0, uHist_modes, pHist, uu_modes_mean, uu_mean, p_mean, res, uMAP_modes, uMAP, pMAP, Hess, evals, evecs, h, h_modes, Dfuh, D2fuh, IVal]
	pickle.dump(data, open("save.p", "wb"))
	
	
	# plots in (log) permeability space
	plt.figure(1)
	plt.plot(x, k, 'k', linewidth=3.0)
	plt.ion()
	for u_m in uHist_modes:
		plt.plot(x, np.exp(evalmodes(u_m, x)))

	plt.plot(x, np.exp(uu_mean), linewidth=3.0)

	plt.figure(2)
	plt.subplot(2, 1, 1)
	# "base" state
	plt.show()


	plt.plot(x, p_mean, linewidth=3.0)

	plt.plot(x, pHist.T, 'k--')
	plt.plot(x[x0_ind], y, 'r.', markersize=15.0)
	plt.plot(x, p, 'g', linewidth=3.0)
	plt.plot(x, pMAP, 'y', linewidth=3.0)

	plt.subplot(2, 1, 2)
	n, bins, patches = plt.hist(IVal)
	plt.axvline(Ifnc_var(uMAP_modes), color = 'y')
	plt.axvline(Ifnc_var(u_modes), color='g')
	plt.axvline(Ifnc_var(uu_modes_mean), color='b')

	plt.figure(3)
	plt.plot(x, p, 'g')
	plt.plot(x[x0_ind], y, 'r.', markersize=15.0)
	plt.plot(x, p_mean, 'r')
	plt.plot(x, pMAP, 'k')
	
	plt.figure(4)
	plt.plot(x, uu_mean, 'r')
	plt.plot(x, u, 'g')
	plt.plot(x, uMAP, 'k')

	plt.figure(5)
	plt.plot(x, pMAP, 'b')
	plt.plot(x, p, 'g')
	plt.plot(x, pMAP + Dfuh, 'r')
	plt.plot(x, pMAP + Dfuh + 0.5*D2fuh, 'y')
	temp = analyseCostFnc(x, uMAP, uMAP_modes, g, x0_ind, y, gamma, pplus, pminus, Ifnc, Lhd, 0.1*evecs[:, 0])
	"""