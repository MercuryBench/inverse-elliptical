from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
import math
from math import e
import sys 
sys.path.append('..')
import mapOnRectangle as mor
from fwdProblem import *
from invProblem2d import *
from rectangle import *
import smtplib
from email.mime.text import MIMEText

rect = Rectangle((0,0), (1,1), resol=7)
gamma = 0.01

def myUTruth(x, y):
		"""if x[0] <= 0.5 +tol  and x[0] >= 0.45 - tol and x[1] <= 0.5+tol:
			return -4
		elif x[0] <= 0.5+tol and x[0] >= 0.45 - tol and x[1] >= 0.6 - tol:
			return -4
		elif x[0] <= 0.75 + tol and x[0] >= 0.7 - tol and x[1] >= 0.2 - tol and x[1] <= 0.8+tol:
			return 2
		else:
			return 0"""
		#if x.ndim == 1 and x.shape[0] == 2:
		return  -4/log10(e)*np.logical_or(np.logical_and(np.logical_and(x < 0.5, x >= 0.4375), y < 0.5), np.logical_and(np.logical_and( x< 0.5 , x >= 0.4375) ,y >= 0.625 - tol)) + 2/log10(e)*np.logical_and(np.logical_and(x < 0.8125, x >= 0.75), np.logical_and(y >= 0.1875, y < 0.75)) + 0

def myUTruth2(x, y):
	return np.logical_and(np.logical_and((x-1)**2 + y**2 <= 0.65**2, (x-1)**2 + y**2 > 0.55**2), x<0.85)*-2/log10(e) + np.sin(4*x)*0.5/log10(e) - 4/log10(e)*np.logical_and(np.logical_and( x< 0.5 , x >= 0.4375) ,y >= 0.625 - tol) 
	
def u_D_term(x, y):
	return np.logical_and(x >= 0.5, y <= 0.625)*2.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] >= 0.6-tol and x[1] <= 0.5:
			return True
		elif x[0] <= 10**-8:
			return True
		else:
			return False

f = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)# (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
m1 = GeneralizedGaussianWavelet2d(rect, 0.1, 1.5, 7)
invProb = inverseProblem(fwd, m1, gamma)


N_obs = 500
obspos = np.random.uniform(0, 1, (2, N_obs))
obspos = [obspos[0,:], obspos[1, :]]
invProb.obspos = obspos

#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
#uTruth = m1.sample()
uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth2(x, y))

obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.plotSolAndLogPermeability(uTruth, obs=obs, save="ground_truth.png")
invProb.obs = obs
#u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
#uOpt = invProb.find_uMAP(u0, nit=3,  method='BFGS', adjoint=False)
start=time.time()
uOpt_adj = invProb.find_uMAP(u0, nit=200, method = 'BFGS', adjoint=True, version=0)
end=time.time()
print("Optimization took " + str(end-start) + " seconds")
invProb.plotSolAndLogPermeability(uOpt_adj, obs=obs)

"""
u = uOpt_adj
h = (m1.sample()) - uOpt_adj
invProb.plotSolAndLogPermeability(u+h, obs=obs)

Fu = invProb.Ffnc(u)
Fu_ = invProb.Ffnc(u, pureFenicsOutput=True)

vtkfile = File('solution.pvd')
vtkfile << Fu_

f = invProb.fwd
V = f.V
mesh = f.mesh


kappa = mor.mapOnRectangle(invProb.rect, "handle", lambda x,y: np.exp(u.handle(x,y)))
kappa1 = mor.mapOnRectangle(invProb.rect, "handle", lambda x,y: np.exp(u.handle(x,y))*h.handle(x,y))
if isinstance(kappa, mor.mapOnRectangle):
	k = morToFenicsConverter(kappa, invProb.fwd.mesh, invProb.fwd.V)

if isinstance(kappa1, mor.mapOnRectangle):
	k1 = morToFenicsConverter(kappa1, invProb.fwd.mesh, invProb.fwd.V)


w = TrialFunction(V)
v = TestFunction(V)
L = -k1*dot(grad(Fu_), grad(v))*dx
a = k*dot(grad(w), grad(v))*dx

wSol = Function(V)
solve(a == L, wSol, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))
vals = np.reshape(wSol.compute_vertex_values(), (2**invProb.rect.resol+1, 2**invProb.rect.resol+1))
Dp = mor.mapOnRectangle(invProb.rect, "expl", vals[0:-1,0:-1])
vtkfile = File('solution_temp.pvd')
vtkfile << wSol

DG_of_u_h = Dp.handle(obspos[0], obspos[1])

discrepancy = obs - Fu.handle(obspos[0], obspos[1])
primalDPhi = -1.0/(gamma**2)*np.dot(discrepancy, DG_of_u_h)
weights = -discrepancy/gamma**2

wtilde = TrialFunction(V)
v = TestFunction(V)
L2 = Constant(0)*v*dx
a2 = a

A2, b2 = assemble_system(a2, L2, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))


for l, weight in enumerate(weights):
	d = PointSource(V, Point(obspos[0][l], obspos[1][l]), weight)
	d.apply(b2)

wtildeSol = Function(V)
solve(A2, wtildeSol.vector(), b2)
vtkfile = File('solution_temptilde.pvd')
vtkfile << wtildeSol


dualDPhi = -assemble(k1*dot(grad(Fu_),grad(wtildeSol))*dx)
fnc = project(k*dot(grad(Fu_), grad(wtildeSol)), V)
vtkfile = File('fnc.pvd')
vtkfile << fnc
hh = morToFenicsConverter(h, invProb.fwd.mesh, invProb.fwd.V)
valsfnc = np.reshape(fnc.compute_vertex_values(), (2**f.rect.resol+1, 2**f.rect.resol+1))
morfnc = mor.mapOnRectangle(f.rect, "expl", valsfnc[0:-1,0:-1]) #cut vals to fit in rect grid


DPhi_dual = invProb.DPhi_adjoint_vec_wavelet(u)
def unitwave(N, l):
	temp = np.zeros((N, ))
	temp[l] = 1
	return mor.mapOnRectangle(f.rect, "wavelet", packWavelet(temp))

"""
"""



print(primalDPhi)
print(invProb.DPhi(u, h))
print(dualDPhi)
print(invProb.DPhi_adjoint(u, h))


# trying to improve DPhi_adjoint
permObspos = kappa.handle(obspos[0], obspos[1])

l1 = [ind for ind in range(len(permObspos)) if permObspos[ind] > 200]
l2 = [ind for ind in range(len(permObspos)) if permObspos[ind] <= 200]

weights1 = weights[l1]
weights2 = weights[l2]
pos1_x = obspos[0][l1]
pos1_y = obspos[1][l1]
pos2_x = obspos[0][l2]
pos2_y = obspos[1][l2]
pos3_x = obspos[0][:]
pos3_y = obspos[1][:]

wtilde = TrialFunction(V)
v = TestFunction(V)
L2 = Constant(0)*v*dx
a2 = k*dot(grad(w), grad(v))*dx


A2_1, b2_1 = 	assemble_system(a2, L2, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))
A2_2, b2_2 = 	assemble_system(a2, L2, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))
A2_3, b2_3 = 	assemble_system(a2, L2, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))

for l, weight in enumerate(weights1):
	d = PointSource(V, Point(pos1_x[l], pos1_y[l]), weight)
	d.apply(b2_1)

for l, weight in enumerate(weights2):
	d = PointSource(V, Point(pos2_x[l], pos2_y[l]), weight)
	d.apply(b2_2)

for l, weight in enumerate(weights):
	d = PointSource(V, Point(pos3_x[l], pos3_y[l]), weight)
	d.apply(b2_3)


	
wtildeSol1 = Function(V)
solve(A2_1, wtildeSol1.vector(), b2_1)
wtildeSol2 = Function(V)
solve(A2_2, wtildeSol2.vector(), b2_2)
wtildeSol3 = Function(V)
solve(A2_3, wtildeSol3.vector(), b2_3)





-assemble(k1*dot(grad(Fu_),grad(wtildeSol1))*dx)-assemble(k1*dot(grad(Fu_),grad(wtildeSol2))*dx)"""






"""m2 = GaussianFourier2d(rect, np.zeros((11,11)), 1.0, 1.0)
invProb2 = inverseProblem(fwd, m2, gamma, obspos, obs)
u0_fourier = mor.mapOnRectangle(rect, "fourier", np.zeros((11,11)))
uOpt_fourier = invProb2.find_uMAP(u0_fourier, nit=100, nfev=100)"""

#uList, uListUnique, PhiList = invProb.randomwalk_pCN(u0, 1000)
#u_new, u_new_mean, us, vals, vals_mean = invProb.EnKF(obs, 50, KL=False, N = 10, beta = 0.05)
#plt.figure(); plt.semilogy(PhiList)
#invProb.plotSolAndLogPermeability(uList[-1])
"""plt.figure();
plt.semilogy(vals)
plt.semilogy(vals_mean, 'r', linewidth=4)
invProb.plotSolAndLogPermeability(u_new_mean)"""


"""Is = []
hvals = np.linspace(-1,1,200)

for k in range(16):
	d = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(16, k)))
	Is.append([invProb.I(u_new_mean + d*h) for h in hvals])

plt.figure()
for k in range(16):
	plt.subplot(4,4,k+1)
	plt.plot(hvals, Is[k])"""
