from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
import math
import sys 
sys.path.append('..')
import mapOnRectangle as mor
from fwdProblem import *
from invProblem2d import *
from rectangle import *

rect = Rectangle((0,0), (1,1), resol=6)
gamma = 0.01

N_obs = 50


def u_D_term(x, y):
	return np.logical_and(x >= 0.5, y <= 0.6)*2.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] >= 0.6-tol and x[1] <= 0.5:
			return True
		elif x[0] <= 10**-8:
			return True
		else:
			return False

f = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
m1 = GeneralizedGaussianWavelet2d(rect, 1, 0.5, 5)
invProb = inverseProblem(fwd, m1, gamma)

Nmodes = len(unpackWavelet(m1.mean))
temp = np.zeros((Nmodes,))
temp[1] = -1.3
temp[2] = 0.7
temp[4] = 0.5
temp[5] = 0.6
thetaTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp))

obspos = [np.random.uniform(0, 1, (N_obs,)), np.random.uniform(0, 1, (N_obs,))]
invProb.obspos = obspos
uTruth = invProb.Ffnc(thetaTruth)
uTruth_ = invProb.Ffnc(thetaTruth, pureFenicsOutput=True)
obs = invProb.Gfnc(thetaTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.plotSolAndLogPermeability(thetaTruth); plt.title("thetaTruth")

Nmodes = len(unpackWavelet(thetaTruth.waveletcoeffs))

temp = np.zeros((Nmodes,))
temp[1] = -1
temp[2] = 0.9
temp[4] = 0.9
temp[6] = -0.2
temp[9] = -0.3
thetabar = mor.mapOnRectangle(rect, "wavelet", packWavelet(temp)); 
invProb.plotSolAndLogPermeability(thetabar, obs=obs); plt.title("thetabar")
temp = np.zeros((Nmodes,))
temp[3] = 0.5
h = thetaTruth - thetabar

theta = thetabar + h
invProb.plotSolAndLogPermeability(theta, obs=obs)

Du_h = invProb.DFfnc(thetabar, h)

u_thetabar = invProb.Ffnc(thetabar)
u_thetabar_ = invProb.Ffnc(thetabar, pureFenicsOutput=True)
u_theta = invProb.Ffnc(theta)


D2u_h = invProb.D2Ffnc(thetabar, h, h)

# optical check for approximation in observation space (Fu, Fu+DFu(h), Fu+DFu(h)+0.5*D2Fu[h,h], Fv) where v = u + h

plt.figure();
plt.subplot(4, 1, 1)
plt.contourf(u_thetabar.values)
plt.subplot(4, 1, 2)
plt.contourf(u_thetabar.values + Du_h.values)
plt.subplot(4, 1, 3)
plt.contourf(u_thetabar.values + Du_h.values + 0.5*D2u_h.values)
plt.subplot(4, 1, 4)
plt.contourf(u_theta.values)

# computational check

print("norm Fu-Fv: " + str(np.sum((u_thetabar.values-u_theta.values)**2)))

print("norm Fu+DFu-Fv: " + str(np.sum((u_thetabar.values+Du_h.values-u_theta.values)**2)))

print("norm Fu+DFu+0.5*D2Fu-Fv: " + str(np.sum((u_thetabar.values+Du_h.values+0.5*D2u_h.values-u_theta.values)**2)))

# optical check for approximation of I(u): (I(u), I(u)+DI(u)(h), I(u)+DI(u)(h)+D2I(u)[h,h], DI(v))
#wtilde = fwd.solveWithDiracRHS(k, ws, xs, pureFenicsOutput=False)
hfactor = np.linspace(-1, 1, 101)
invProb.obs = obs
Ivals = np.array([invProb.I(thetabar+h*hfac) for hfac in hfactor])
Ivals_D = invProb.I(thetabar) + hfactor*invProb.DI(thetabar, h)
Ivals_D2 = invProb.I(thetabar) + hfactor*invProb.DI(thetabar, h) + 0.5*hfactor**2*invProb.D2I(thetabar, h, h)

plt.figure();
plt.plot(hfactor, Ivals, '.-')

plt.plot(hfactor, Ivals_D, 'r.-')
plt.plot(hfactor, Ivals_D2, 'g.-')


mesh = invProb.fwd.mesh
V = invProb.fwd.V
kappa = mor.mapOnRectangle(invProb.rect, "handle", lambda x,y: np.exp(thetabar.handle(x,y)))
kappa1 = mor.mapOnRectangle(invProb.rect, "handle", lambda x,y: np.exp(thetabar.handle(x,y))*h.handle(x,y))
if isinstance(kappa, mor.mapOnRectangle):
	k = morToFenicsConverter(kappa, invProb.fwd.mesh, invProb.fwd.V)
if isinstance(kappa1, mor.mapOnRectangle):
	k1 = morToFenicsConverter(kappa1, invProb.fwd.mesh, invProb.fwd.V)

w = TrialFunction(V)
v = TestFunction(V)
L = -k1*dot(grad(u_thetabar_), grad(v))*dx
a = k*dot(grad(w), grad(v))*dx

#A, b = assemble_system(a, L, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))

wSol = Function(V)
solve(a == L, wSol, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))
vals = np.reshape(wSol.compute_vertex_values(), (2**invProb.rect.resol+1, 2**invProb.rect.resol+1))
Dp = mor.mapOnRectangle(invProb.rect, "expl", vals[0:-1,0:-1])
vtkfile = File('solution_temp.pvd')
vtkfile << wSol

DG_of_u_h = Dp.handle(obspos[0], obspos[1])
discrepancy = obs - u_thetabar.handle(obspos[0], obspos[1])
value0 = -1.0/(gamma**2)*np.dot(discrepancy, DG_of_u_h)

wtilde = TrialFunction(V)
v = TestFunction(V)
L2 = Constant(0)*v*dx
a2 = a

A2, b2 = assemble_system(a2, L2, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))

weights = -discrepancy/gamma**2
for k, weight in enumerate(weights):
	d = PointSource(V, Point(obspos[0][k], obspos[1][k]), weight)
	d.apply(b2)

wtildeSol = Function(V)
solve(A2, wtildeSol.vector(), b2)
vtkfile = File('solution_temp2.pvd')
vtkfile << wtildeSol
plot(wtildeSol)

value1 = -assemble(k1*dot(grad(u_thetabar_),grad(wtildeSol))*dx)

"""
#DD = invProb.gradientI_adjoint_wavelet(u)
invProb.DPhi(u, h)

p = invProb.Ffnc(u)
p_ = invProb.Ffnc(u, pureFenicsOutput=True)
Gvals = p.handle(obspos[0], obspos[1])
discrepancy = obs-Gvals


F_u = invProb.Ffnc(u, pureFenicsOutput=True)
kappa = mor.mapOnRectangle(invProb.rect, "handle", lambda x,y: np.exp(u.handle(x,y)))
kappa1 = mor.mapOnRectangle(invProb.rect, "handle", lambda x,y: np.exp(u.handle(x,y))*h.handle(x,y))
		

if isinstance(kappa, mor.mapOnRectangle):
	k = morToFenicsConverter(kappa, invProb.fwd.mesh, invProb.fwd.V)
if isinstance(kappa1, mor.mapOnRectangle):
	k1 = morToFenicsConverter(kappa1, invProb.fwd.mesh, invProb.fwd.V)

# primal 
u = TrialFunction(invProb.fwd.V)
v = TestFunction(invProb.fwd.V)
L = - k1*dot(grad(F_u),grad(v))*dx
a = k*dot(grad(u), grad(v))*dx
uSol = Function(invProb.fwd.V)
solve(a == L, uSol, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))
vals = np.reshape(uSol.compute_vertex_values(), (2**invProb.rect.resol+1, 2**invProb.rect.resol+1))
Dp = mor.mapOnRectangle(invProb.rect, "expl", vals[0:-1,0:-1])


DG_of_u_h = Dp.handle(obspos[0], obspos[1])
value0 = -1.0/(gamma**2)*np.dot(discrepancy, DG_of_u_h)

# dual
u = TrialFunction(invProb.fwd.V)
v = TestFunction(invProb.fwd.V)
diracs = [PointSource(invProb.fwd.V, Point(obspos[0][k], obspos[1][k]), discrepancy[k]) for k in range(len(discrepancy))]
L = Constant(0)*v*dx
a = k*dot(grad(u), grad(v))*dx
A, b = assemble_system(a, L, DirichletBC(invProb.fwd.V, Constant(0), invProb.fwd.boundary_markers, 1))
for d in diracs:
	d.apply(b)

uSolDual = Function(invProb.fwd.V)
solve(A, uSolDual.vector(), b)

intvalue = k1*dot(grad(uSolDual), grad(p_))*dx"""

"""# primal method
k = mor.mapOnRectangle(rect, "expl", np.exp(uu.values))
k1 = k*h
y = Fu
w = fwd.solveWithHminus1RHS(k, k1, y, pureFenicsOutput=False) # solves -div(k*nabla(y1)) = div(k1*nabla(y)) for y1	
value1 = -1/gamma**2 * sum((obs-Fu.handle(obspos[0], obspos[1]))*w.handle(obspos[0], obspos[1]))

mesh = invProb.fwd.mesh
V = invProb.fwd.V



#dual method
ws = obs - Fu.handle(obspos[0], obspos[1])
xs = zip(obspos[0], obspos[1])
wtilde_ = fwd.solveWithDiracRHS(k, ws, xs, pureFenicsOutput=True)
wtilde = fwd.solveWithDiracRHS(k, ws, xs, pureFenicsOutput=False) # solves -div(k*nabla(y)) = sum_i w_i*dirac_{x_i}
Fu_ = invProb.Ffnc(uu, pureFenicsOutput=True)
value2 = assemble(morToFenicsConverter(k1, invProb.fwd.mesh, invProb.fwd.V)*dot(grad(Fu_), grad(wtilde_))*dx)"""


