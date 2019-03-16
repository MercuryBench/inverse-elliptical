import numpy as np
from scipy import integrate
import time 
import matplotlib.pyplot as plt

x = np.linspace(0, 2, num=100)
y = x
z = np.zeros_like(y)
dx = x[1]-x[0]
st = time.time()
for m in range(100):
	for n, xval in enumerate(x):
		z[n] = np.trapz(x[0:n], y[0:n], dx = dx)
et = time.time()
for m in range(100):
	z2 = np.concatenate((np.array([0]), integrate.cumtrapz(y, x, dx=dx)))
et2 = time.time()


print(et-st)
print(et2-et)
