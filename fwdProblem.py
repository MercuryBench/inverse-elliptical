from __future__ import division
import numpy as np
import mapOnInterval as moi

class linEllipt():
	# model: -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
	def __init__(self, g, pplus, pminus):
		self.g = g
		self.pplus = pplus
		self.pminus = pminus

	def solve(self, x, k, returnC = False):
		# solves -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
		kinv = moi.mapOnInterval("handle", lambda x: 1/k.handle(x))
		I_1 = moi.integrate(x, kinv)
		I_2 = moi.integrate(x, self.g)
		I_2timeskinv = moi.mapOnInterval("expl", I_2/k.values)
		I_3 = moi.integrate(x, I_2timeskinv)
		C = (self.pplus - self.pminus + I_3[-1])/(I_1[-1])
		p = moi.mapOnInterval("expl", -I_3 + C*I_1 + self.pminus)
		if returnC:
			return p, C
		else:
			return p
