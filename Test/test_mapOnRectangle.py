from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from mapOnRectangle import *

m1 = mapOnRectangle((0, 1), (2, 5), "wavelet", packWavelet(np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0])), resol=4)
mat = np.zeros((5,5))
mat[1,3] = 1
mat[0,2] = 0.5
m2 = mapOnRectangle((0, 0), (100, 10), "fourier", mat, resol=4)
m3 = mapOnRectangle((0, 0), (100, 10), "handle", lambda x, y: 1/100*x*(y**3) + x**2, resol=6)

plt.figure(); plt.ion();
plt.subplot(3, 1, 1);
X1,Y1 = m1.getXYmeshgrid()
plt.contourf(X1, Y1, m1.values); plt.colorbar()
plt.subplot(3, 1, 2);
X2,Y2 = m2.getXYmeshgrid()
plt.contourf(X2, Y2, m2.values); plt.colorbar()
plt.subplot(3, 1, 3);
X3,Y3 = m3.getXYmeshgrid()
plt.contourf(X3, Y3, m3.handle(X3, Y3)); plt.colorbar()
plt.show()
plt.matshow(m3.values)

# test transformation:
wcm3 = m3.waveletcoeffs
m4 = mapOnRectangle((0, 1), (2, 5), "wavelet", wcm3, resol=6)
m5 = mapOnRectangle((0, 1), (2, 5), "expl", m4.values, resol=6)
plt.figure()
plt.subplot(2, 1, 1)
plt.contourf(X3, Y3, m4.values); plt.colorbar()
plt.subplot(2, 1, 2)
plt.contourf(X3, Y3, m5.values); plt.colorbar()
