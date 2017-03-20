from __future__ import division
import numpy as np
import mapOnInterval as moi

class linEllipt():
	# model: -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
	def __init__(self, g, pplus, pminus):
		self.g = g
		self.pplus = pplus
		self.pminus = pminus

	def solve(self, x, k, g = None, pplus=None, pminus=None, returnC = False):
		# solves -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
		# if g == None, we take self.g, else we take this as the right hand side (same for pplus and pminus)
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
		I_2timeskinv = moi.mapOnInterval("expl", I_2/k.values)
		I_3 = moi.integrate(x, I_2timeskinv)
		C = (pplus - pminus + I_3[-1])/(I_1[-1])
		p = moi.mapOnInterval("expl", -I_3 + C*I_1 + pminus)
		if returnC:
			return p, C
		else:
			return p
