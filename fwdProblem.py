from __future__ import division
import numpy as np
import mapOnInterval as moi
import time
from measures import  *
from math import pi
import matplotlib.pyplot as plt
import scipy

class linEllipt():
	# model: -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
	def __init__(self, g, pplus, pminus):
		self.g = g
		self.pplus = pplus
		self.pminus = pminus

	def solve(self, x, k, g = None, pplus=None, pminus=None, returnC = False, moiMode = False):
		# solves -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
		# if g == None, we take self.g, else we take this as the right hand side (same for pplus and pminus)
		if moiMode == True:
			kinv = moi.mapOnInterval("handle", lambda x: 1/k.handle(x))
			I_1 = moi.integrate(x, kinv)
			if g == None:
				I_2 = moi.integrate(x, self.g)
			else:
				I_2 = moi.integrate(x, g)	
			if pplus == None:
				pplus = self.pplus
			if pminus == None:
				pminus = self.pminus
			I_2timeskinv = moi.mapOnInterval("expl", I_2.values/k.values)
			I_3 = moi.integrate(x, I_2timeskinv)
			C = (pplus - pminus + I_3.values[-1])/(I_1.values[-1])
			p = moi.mapOnInterval("expl", -I_3.values + C*I_1.values + pminus)
		else: # non-moi mode:
			kvals = k.values
			if g == None:
				g = self.g
			if pplus == None:
				pplus = self.pplus
			if pminus == None:
				pminus = self.pminus
			gvals = g.values
			I_1 = np.concatenate((np.array([0]), scipy.integrate.cumtrapz(1/kvals, x)))
			I_2 = np.concatenate((np.array([0]), scipy.integrate.cumtrapz(gvals, x)))
			I_3 = np.concatenate((np.array([0]), scipy.integrate.cumtrapz(I_2/kvals, x)))
			C = (pplus - pminus + I_3[-1])/(I_1[-1])
			p = moi.mapOnInterval("expl", -I_3 + C*I_1 + pminus)
		
		if returnC:
			return p, C
		else:
			return p

if __name__ == "__main__":
		x = np.linspace(0, 1, 512)
	
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
		k0 = moi.mapOnInterval("expl", np.exp(u0.values))
		
		lE = linEllipt(g, pplus, pminus)
		
		st = time.time()
		for m in range(100):
			p = lE.solve(x, k0, moiMode = True)
			
		et = time.time()
		print(et-st)
		st = time.time()
		for m in range(100):
			p2 = lE.solve(x, k0, moiMode = False)
		et = time.time()
		print(et-st)
		plt.show()
