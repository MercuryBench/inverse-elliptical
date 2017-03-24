from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp
from fwdProblem import *
from measures import *
import mapOnInterval as moi
import pickle
import time, sys

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


class inverseProblem():
	def __init__(self, fwd, prior, gamma, obsind=None, obs=None):
		# need: type(fwd) == fwdProblem, type(prior) == measure
		self.fwd = fwd
		self.prior = prior
		self.obsind = obsind
		self.obs = obs
		self.gamma = gamma
		
	# Forward operators and their derivatives:	
	def Ffnc(self, x, u, g=None, pplus=None, pminus=None): # F is like forward, but uses logpermeability instead of permeability
		# if g == None, then we take the default right hand side
		perm = moi.mapOnInterval("handle", lambda x: np.exp(u.handle(x)))
		return self.fwd.solve(x, perm, g, pplus, pminus)
	def DFfnc(self, x, u, h):
		p = self.Ffnc(x, u)
		h_arr = h.values
		u_arr = u.values
		pprime = moi.differentiate(x, p)
		tempfnc = moi.mapOnInterval("expl", h.values*np.exp(u.values)*pprime.values)
		rhs = moi.differentiate(x, tempfnc)
		p1 = self.Ffnc(x, u, rhs, 0, 0)
		return p1
	def D2Ffnc(self, x, u, h1, h2): 
		p = self.Ffnc(x, u)
		#h1_arr = h1.values
		#h2_arr = h2.values
		#u_arr = u.values
		pprime = moi.differentiate(x, p)
		tempfnc1 = moi.mapOnInterval("expl", h1.values*np.exp(u.values)*pprime.values)
		rhs1 = moi.differentiate(x, tempfnc1)
		#rhs1 = moi.differentiate(x, h1_arr*np.exp(u_arr)*pprime_arr)
		#rhs2 = moi.differentiate(x, h2_arr*np.exp(u_arr)*pprime_arr)
		tempfnc2 = moi.mapOnInterval("expl", h2.values*np.exp(u.values)*pprime.values)
		rhs2 = moi.differentiate(x, tempfnc2)
		p1 = self.Ffnc(x, u, rhs1, 0, 0)
		p2 = self.Ffnc(x, u, rhs2, 0, 0)
		p1prime = moi.differentiate(x, p1)
		p2prime = moi.differentiate(x, p2)
		tempfnc11 = moi.mapOnInterval("expl", np.exp(u.values)*(h1.values*p2prime.values + h2.values*p1prime.values))
		tempfnc22 = moi.mapOnInterval("expl", np.exp(u.values)*(h1.values*h2.values*pprime.values))
		rhs11 = moi.differentiate(x, tempfnc11)
		rhs22 = moi.differentiate(x, tempfnc22)
		#rhs11 = moi.differentiate(x, np.exp(u_arr)*(h1_arr*p2prime + h2_arr*p1prime))
		#rhs11 = moi.differentiate(x, np.exp(u_arr)*(h1_arr*h2_arr*pprime_arr))
		p22 = self.Ffnc(x, u, rhs22, 0, 0)
		p11 = self.Ffnc(x, u, rhs11, 0, 0)
		return moi.mapOnInterval("expl", p11.values + p22.values)
	
	def Gfnc(self, x, u):
		if self.obsind == None:
			raise ValueError("obsind need to be defined")
		p = self.Ffnc(x, u)
		obs = p.handle(x)[self.obsind]
		return obs
	def DGfnc(self, x, u, h):
		Dp = DFfnc(x, u, h)
		return Dp.handle(x)[self.obsind]
	def D2Gfnc(self, x, u, h1, h2):
		D2p = D2Ffnc(x, u, h1, h2)
		return D2p.handle(x)[self.obsind]
	
	def Phi(self, x, u, obs):
		discrepancy = obs-self.Gfnc(x, u)
		return 1/(2*self.gamma**2)*np.dot(discrepancy,discrepancy) 
	def I(self, x, u, obs):
		return self.Phi(x, u, obs) + prior.normpart(u)
	
	def randomwalk(self, uStart, obs, delta, N): # for efficiency, only save modes, not whole function
		u = uStart
		
		if uStart.inittype == "fourier":
			u_modes = uStart.fouriermodes
			uHist = [u_modes]
			for n in range(N):
				v_modes = sqrt(1-2*delta)*u.fouriermodes + sqrt(2*delta)*prior.sample().fouriermodes # change after overloading
				v = moi.mapOnInterval("fourier", v_modes)
				if ip.Phi(x, u, obs) - ip.Phi(x, v, obs) > 1:
					alpha = 1
				else:
					alpha = min(1, exp(ip.Phi(x, u, obs) - ip.Phi(x, v, obs)))
				r = np.random.uniform()
				if r < alpha:
					u = v
					u_modes = v_modes
				uHist.append(u_modes)
			return uHist
		elif uStart.inittype == "wavelet":
			u_coeffs = uStart.waveletcoeffs
			uHist = [u_coeffs]
			for n in range(N):
				v_coeffs = []
				step = prior.sample().waveletcoeffs
				for n, uwc in enumerate(u.waveletcoeffs):
					v_coeffs.append(sqrt(1-2*delta)*uwc + sqrt(2*delta)*step[n])
				v = moi.mapOnInterval("wavelet", v_coeffs)
				if ip.Phi(x, u, obs) - ip.Phi(x, v, obs) > 1:
					alpha = 1
				else:
					alpha = min(1, exp(ip.Phi(x, u, obs) - ip.Phi(x, v, obs)))
				r = np.random.uniform()
				if r < alpha:
					u = v
					u_coeffs = v_coeffs
				uHist.append(u_coeffs)
			return uHist

if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1] == "g":
		x = np.linspace(0, 1, 512)
		gamma = 0.02
		delta = 0.05
	
		# boundary values for forward problem
		# -(k * p')' = g
		# p(0) = pminus
		# p(1) = pplus
		pplus = 2.0
		pminus = 1.0	
		# right hand side of forward problem
		g = moi.mapOnInterval("handle", lambda x: 3.0*x*(1-x))	
		# construct forward problem
		fwd = linEllipt(g, pplus, pminus)
	
		# prior measure:
		alpha = 0.7
		beta = 0.5
		mean = np.zeros((31,))
		prior = GaussianFourier(mean, alpha, beta)
	
		# case 1: random ground truth
		u0 = prior.sample()
		
		# case 2: given ground truth
		J = 9
		num = 2**J
		x = np.linspace(0, 1, 2**(J), endpoint=False)
		gg1 = lambda x: 1 + 2**(-J)/(x**2+2**J) + 2**J/(x**2 + 2**J)*np.cos(32*x)
		g1 = lambda x: gg1(2**J*x)
		gg2 = lambda x: (1 - 0.4*x**2)/(2**(J+3)) + np.sin(7*x/(2*pi))/(1 + x**2/2**J)
		g2 = lambda x: gg2(2**J*x)
		gg3 = lambda x: 3 + 3*(x**2/(2**(2*J)))*np.sin(x/(8*pi))
		g3 = lambda x: gg3(2**J*x)
		gg4 = lambda x: (x**2/3**J)*0.1*np.cos(x/(2*pi))-x**3/8**J + 0.1*np.sin(3*x/(2*pi))
		g4 = lambda x: gg4(2**J*x)
		vec1 = g2(x[0:2**(J-5/2)])
		vec2 = g1(x[2**(J-5/2):2**(J-1.5)])
		vec3 = g3(x[2**(J-1.5):2**(J)-2**(J-1.2)])
		vec4 = g4(x[2**(J)-2**(J-1.2):2**(J)])

		f = np.concatenate((vec1, vec2, vec3, vec4))
		u0 = moi.mapOnInterval("expl", f)
		
		k0 = moi.mapOnInterval("handle", lambda x: np.exp(u0.handle(x)))
		plt.figure(1)
		plt.ion()
		plt.plot(x, u0.handle(x))
		plt.show()
	
		# construct solution and observation
		p0 = fwd.solve(x, k0)
		x0_ind = range(10, 490, 10) # observation indices
		obs = p0.values[x0_ind] + np.random.normal(0, gamma, (len(x0_ind),))
		plt.figure(2)
		plt.plot(x, p0.handle(x), 'k')
	
		plt.plot(x[x0_ind], obs, 'r.')
	
		ip = inverseProblem(fwd, prior, gamma, x0_ind, obs)
	
		uHist = ip.randomwalk(prior.sample(), obs, delta, 150)
		plt.figure(3)
	
		for uh in uHist:
			plt.plot(x, moi.evalmodes(uh, x))
		uHist_mean = moi.mapOnInterval("fourier", np.mean(uHist, axis=0))
		pHist_mean = ip.Ffnc(x, uHist_mean)
		pStart = ip.Ffnc(x, moi.mapOnInterval("fourier", uHist[0]))
	
		plt.plot(x, uHist_mean.handle(x), 'g', linewidth=4)
		plt.plot(x, moi.evalmodes(uHist[0], x), 'r', linewidth=4)
		plt.plot(x, u0.values, 'k', linewidth=4)
	
		plt.figure(2)
		plt.plot(x, pStart.handle(x), 'r')
		plt.plot(x, pHist_mean.handle(x), 'g')
	
		# test gradient calculation
		h = moi.mapOnInterval("handle", lambda x: u0.handle(x) - uHist_mean.handle(x))
		DFuh = ip.DFfnc(x, uHist_mean, h)
		D2Fuh = ip.D2Ffnc(x, uHist_mean, h, h)
		plt.figure(4)
		plt.plot(x, pHist_mean.handle(x), 'g')
		plt.plot(x, p0.handle(x), 'k')
		T1 = moi.mapOnInterval("handle", lambda x: pHist_mean.handle(x) + DFuh.handle(x))
		T2 = moi.mapOnInterval("handle", lambda x: pHist_mean.handle(x) + DFuh.handle(x) + 0.5*D2Fuh.handle(x))
		plt.plot(x, T1.handle(x), 'b')
		plt.plot(x, T2.handle(x), 'm')
		plt.plot(x[x0_ind], obs, 'r.')
		#h = u - uMAP
		#h_modes = u_modes - uMAP_modes 
		#Dfuh = DF_long(x, uMAP, g, pplus, pminus, h)
		#D2fuh = D2F_long(x, uMAP, g, pplus, pminus, h, h)
	else:
		x = np.linspace(0, 1, 512)
		gamma = 0.02
		delta = 0.005
	
		# boundary values for forward problem
		# -(k * p')' = g
		# p(0) = pminus
		# p(1) = pplus
		pplus = 2.0
		pminus = 1.0	
		# right hand side of forward problem
		g = moi.mapOnInterval("handle", lambda x: 3.0*x*(1-x))	
		# construct forward problem
		fwd = linEllipt(g, pplus, pminus)
		
		# prior measure:
		maxJ = 9
		kappa = 100.0
		prior = LaplaceWavelet(kappa, maxJ)
		
		# case 1: random ground truth
		u0 = prior.sample()
		
		# case 2: given ground truth
		J = 9
		num = 2**J
		x = np.linspace(0, 1, 2**(J), endpoint=False)
		gg1 = lambda x: 1 + 2**(-J)/(x**2+2**J) + 2**J/(x**2 + 2**J)*np.cos(32*x)
		g1 = lambda x: gg1(2**J*x)
		gg2 = lambda x: (1 - 0.4*x**2)/(2**(J+3)) + np.sin(7*x/(2*pi))/(1 + x**2/2**J)
		g2 = lambda x: gg2(2**J*x)
		gg3 = lambda x: 3 + 3*(x**2/(2**(2*J)))*np.sin(x/(8*pi))
		g3 = lambda x: gg3(2**J*x)
		gg4 = lambda x: (x**2/3**J)*0.1*np.cos(x/(2*pi))-x**3/8**J + 0.1*np.sin(3*x/(2*pi))
		g4 = lambda x: gg4(2**J*x)
		vec1 = g2(x[0:2**(J-5/2)])
		vec2 = g1(x[2**(J-5/2):2**(J-1.5)])
		vec3 = g3(x[2**(J-1.5):2**(J)-2**(J-1.2)])
		vec4 = g4(x[2**(J)-2**(J-1.2):2**(J)])

		f = np.concatenate((vec1, vec2, vec3, vec4))
		f = f - np.sum(f)*2**(-9) # normalize
		u0 = moi.mapOnInterval("expl", f)
		
		k0 = moi.mapOnInterval("handle", lambda x: np.exp(u0.handle(x)), interpolationdegree=1)
		plt.figure(1)
		plt.ion()
		plt.plot(x, u0.handle(x))
		plt.show()
		
		# construct solution and observation
		p0 = fwd.solve(x, k0)
		x0_ind = range(10, 490, 10) # observation indices
		obs = p0.values[x0_ind] + np.random.normal(0, gamma, (len(x0_ind),))
		plt.figure(2)
		plt.plot(x, p0.handle(x), 'k')
		plt.plot(x[x0_ind], obs, 'r.')
		
		ip = inverseProblem(fwd, prior, gamma, x0_ind, obs)
		
		# possibility 1: sample start
		uStart = prior.sample()
		
		# possibility 2: specific start
		ww = uStart.waveletcoeffs
		for n, w in enumerate(ww):
			ww[n] = np.zeros_like(w)
		ww[2] = np.array([0.2, -0.15, -0.05, 0.1])
		uStart = moi.mapOnInterval("wavelet", ww)
		
		
		uHist = ip.randomwalk(uStart, obs, delta, 450)
		plt.figure(3)
		uHistfnc = []
		pHistfnc = []
		IHist = []
		PhiHist = []
		"""for uh in uHist:
			uhfnc = moi.mapOnInterval("wavelet", uh, interpolationdegree=1)
			pfnc = ip.Ffnc(x, uhfnc)
			#plt.plot(x, uhfnc.handle(x))
			uHistfnc.append(uhfnc)
			pHistfnc.append(pfnc)
			IHist.append(ip.I(x, uhfnc, obs))
			PhiHist.append(ip.Phi(x, uhfnc, obs))"""
		
		uHist_mean_c = []
		for j in range(len(uHist[0])):
			avg = 0
			for k in range(30, len(uHist)):
				avg = avg + uHist[k][j]
			avg = avg/len(uHist)
			uHist_mean_c.append(avg)
			update_progress(j/len(uHist[0]))
		
		uHist_mean = moi.mapOnInterval("wavelet", uHist_mean_c, interpolationdegree=1)
		pHist_mean = ip.Ffnc(x, uHist_mean)
		pStart = ip.Ffnc(x, moi.mapOnInterval("wavelet", uHist[0]))
	
		plt.plot(x, uHist_mean.handle(x), 'g', linewidth=1)
		plt.plot(x, moi.mapOnInterval("wavelet", uHist[0], interpolationdegree=1).handle(x), 'r', linewidth=1)
		plt.plot(x, u0.handle(x), 'k', linewidth=1)
		
		plt.figure(2)
		plt.plot(x, pStart.handle(x), 'r')
		plt.plot(x, pHist_mean.handle(x), 'g')
		plt.plot(x, pHistfnc[-1].handle(x), 'b')
		
		data = [x, gamma, delta, pplus, pminus, maxJ, kappa, u0.values, J, num, f, k0.values, p0.values, x0_ind, obs, uStart.values, uHist, uHist_mean.values, pHist_mean.values, pStart.values]
		str = "save_" + datetime.datetime.now().isoformat('_') + ".p"

		pickle.dump(data, open(str, "wb"))
		
		"""plt.figure()
		for pf in pHistfnc:
			plt.plot(x, pf.handle(x))
		plt.plot(x, pStart.handle(x), 'r', linewidth=4)
		plt.plot(x, pHist_mean.handle(x), 'g', linewidth=4)
		plt.plot(x, p0.handle(x), 'k', linewidth=4)"""
		
		#plt.figure()
		#plt.plot(IHist)
		#plt.plot(PhiHist, 'r')

