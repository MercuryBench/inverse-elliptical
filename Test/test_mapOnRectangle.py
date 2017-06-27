from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from mapOnRectangle import *
from rectangle import *
from measures import *

rect1 = Rectangle((0,1),(2,5),resol=4)

m1 = mapOnRectangle(rect1, "wavelet", packWavelet(np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0])))
mat = np.zeros((5,5))
mat[1,3] = 1
mat[0,2] = 0.5

rect = Rectangle((0, 0), (100, 10), resol=6)
m2 = mapOnRectangle(rect, "fourier", mat)
m3 = mapOnRectangle(rect, "handle", lambda x, y: 1/100*x*(y**3) + x**2)

plt.figure(); plt.ion();
plt.subplot(3, 1, 1);
X1,Y1 = m1.X, m1.Y
plt.contourf(X1, Y1, m1.values); plt.colorbar()
plt.subplot(3, 1, 2);
X2,Y2 = m2.X, m2.Y
plt.contourf(X2, Y2, m2.values); plt.colorbar()
plt.subplot(3, 1, 3);
X3,Y3 = m3.X, m3.Y
plt.contourf(X3, Y3, m3.handle(X3, Y3)); plt.colorbar()
plt.show()
plt.matshow(m3.values)

# test transformation:
wcm3 = m3.waveletcoeffs
m4 = mapOnRectangle(rect, "wavelet", wcm3)
m5 = mapOnRectangle(rect, "expl", m4.values)
plt.figure()
plt.subplot(2, 1, 1)
plt.contourf(X3, Y3, m4.values); plt.colorbar()
plt.subplot(2, 1, 2)
plt.contourf(X3, Y3, m5.values); plt.colorbar()


rect2 = Rectangle((0,0), (2, 1), resol = 2)
p = GeneralizedGaussianWavelet2d(rect2, 1.0, 0.0, 2)
f1 = p.sample() 
f2 = mapOnRectangle(rect2, "expl", f1.values)
f3 = mapOnRectangle(rect2, "wavelet", f2.waveletcoeffs)
f4 = mapOnRectangle(rect2, "handle", f3.handle)

g1 = mapOnRectangle(rect2, "handle", lambda x, y: 10*x + y)
g2 = mapOnRectangle(rect2, "expl", g1.values)
g3 = mapOnRectangle(rect2, "handle", g2.handle)

plt.figure()
plt.contourf(g1.X, g1.Y, g1.values)
