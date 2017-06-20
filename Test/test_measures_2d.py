from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from measures import *
from rectangle import *

rect = Rectangle((0,1),(2,50),resol=6)

m1 = GaussianFourier2d(rect, np.zeros((13,13)), 1.0, 1.0)
u1 = m1.sample()

plt.figure(); plt.ion();
X1,Y1 = u1.X, u1.Y
plt.contourf(X1, Y1, u1.values, 50); plt.colorbar()
plt.show()

m2 = GeneralizedGaussianWavelet2d(rect, 1.0, 0.5, 5)
u2 = m2.sample()
plt.figure(); plt.ion();
X2,Y2 = u2.X, u2.Y
plt.contourf(X2, Y2, u2.values, 50); plt.colorbar()
plt.show()
