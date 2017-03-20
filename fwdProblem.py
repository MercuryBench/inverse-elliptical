from __future__ import division
import numpy as np
import mapOnInterval as moi

class linEllipt():
	# model: -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
	def __init__(self, g, pplus, pminus):
		self.g = g
		self.pplus = pplus
		self.pminus = pminus

	def solve(self, x, k):
		# solves -(k*p')' = g, with p(0) = pminus, p(1) = pplus, for p
		I_1 = moi.integrate(x, 1.0/k)
		I_2 = moi.integrate(x, self.g)
		I_3 = moi.integrate(x, I_2/k)
		C = (self.pplus - self.pminus + I_3[-1])/(I_1[-1])
		p = -I_3 + C*I_1 + self.pminus
		return p, C
