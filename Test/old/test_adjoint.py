from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from fwdProblem import *
from measures import *
from haarWavelet2d import *


rect = Rectangle((0,0), (180,78), resol=7)

f = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-40)**2 + (y-20)**2) < 400)*(-1.0))

def u_D_term(x, y):
	return np.logical_and(x >= 80-tol, y <= 40)*2.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] >= 90-tol and x[1] <= 30:
			return True
		elif x[0] <= 10**-8 and x[1] >= 10 and x[1] <= 40:
			return True
		else:
			return False


fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)

m1 = GeneralizedGaussianWavelet2d(rect, 1.0, 1.5, 5)
u1 = m1.sample()
plt.figure();
plt.contourf(u1.values)
plt.colorbar()
k1 = mor.mapOnRectangle(rect, "handle", lambda x,y: np.exp(u1.handle(x, y)))
k1 = mor.mapOnRectangle(rect, "handle", lambda x,y: 0*x + 1)
F1_ = fwd.solve(k1, pureFenicsOutput=True)
F1 = fwd.solve(k1)
vtkfile = File('solution_sandbox.pvd')
vtkfile << F1_

plt.figure(); plt.ion()
plt.contourf(F1.X, F1.Y, F1.values, 60); plt.colorbar()
plt.show()

value1 = F1_(60, 40)
# now adjoint problem


fwd2 = linEllipt2dRectangle(rect, f, Constant(0), boundary_D_boolean)
F2 = fwd2.solveWithDiracRHS(k1, [1], [[60, 40]], pureFenicsOutput=False) # solves -div(k*nabla(y)) = sum_i w_i*dirac_{x_i}
F2_ = fwd2.solveWithDiracRHS(k1, [1], [[60, 40]], pureFenicsOutput=True) # solves -div(k*nabla(y)) = sum_i w_i*dirac_{x_i}
plt.figure(); plt.ion()
plt.contourf(F2.X, F2.Y, F2.values, 60); plt.colorbar()
plt.show()

boundary_markers = FacetFunction("size_t", fwd2.mesh)
class BoundaryD2(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and near(x[0], 0, tol) and (x[1] >= 10-tol and x[1] <= 40+tol)

bd2 = BoundaryD2()
bd2.mark(boundary_markers, 0)

ds2 = Measure("ds", domain=fwd2.mesh, subdomain_data=boundary_markers)

intfnc1 = Expression("pow((x[0]-40), 2) + pow((x[1]-20), 2) < 400 ? -1 : 0", degree=2)
int1 = intfnc1*F2_ * dx
n = FacetNormal(fwd2.mesh)
k1_fenics = morToFenicsConverter(k1, fwd2.mesh, fwd2.V)
value2 = assemble(int1) - 2*assemble(k1_fenics*dot(grad(F2_), n)*ds2(0))
