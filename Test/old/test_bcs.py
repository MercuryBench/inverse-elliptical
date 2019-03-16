from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
import sys 
sys.path.append('..')
from fwdProblem import *
from measures import *
import mapOnRectangle as mor
from rectangle import *

resol=8

rect = Rectangle((0,0), (180,78), resol=resol)

mesh = RectangleMesh(Point(0,0), Point(180,78), 2**resol, 2**resol)
V = FunctionSpace(mesh, 'P', 1)

#f = mor.mapOnRectangle(rect, "handle", lambda x, y: (((x-40)**2 + (y-20)**2) < 400)*(-.01))
#f = morToFenicsConverter(f, mesh, V)
f = Constant(0)

boundary_markers = FacetFunction("size_t", mesh)

def boundary_D_classifier(x): 
	tol = 1E-14
	if x[0] >= 90-tol and x[1] <= 30+tol:
		return "D2"
	elif x[0] <= tol and x[1] >= 10-tol and x[1] <= 40+tol:
		return "D1"
	else:
		return "N"

class BoundaryD1(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and boundary_D_classifier(x) == "D1"
class BoundaryD2(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and boundary_D_classifier(x) == "D2"
class BoundaryN(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and boundary_D_classifier(x) == "N"

BD1 = BoundaryD1()
BD1.mark(boundary_markers, 1)
BD2 = BoundaryD2()
BD2.mark(boundary_markers, 2)
BN = BoundaryN()
BN.mark(boundary_markers, 3)

bcs = [DirichletBC(V, Constant(0.0), boundary_markers, 1), DirichletBC(V, Constant(2.0), boundary_markers, 2)]
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

u = TrialFunction(V)
v = TestFunction(V)

m1 = GeneralizedGaussianWavelet2d(rect, 2.0, 1.5, 3)
u1 = m1.sample()
plt.figure();
plt.ion()
plt.contourf(u1.values)
plt.colorbar()
plt.show()
kappa = mor.mapOnRectangle(rect, "handle", lambda x,y: np.exp(u1.handle(x, y)))
kappa_fenics = morToFenicsConverter(kappa, mesh, V)
a = kappa_fenics*dot(grad(u), grad(v))*dx
L = f*v*dx

A, b = assemble_system(a, L, bcs)

uSol = Function(V)
#solve(A, uSol.vector(), b)
solve(a == L, uSol, bcs)#)bcs[1])
vals = np.reshape(uSol.compute_vertex_values(), (2**rect.resol+1, 2**rect.resol+1))
uSol_mor = mor.mapOnRectangle(rect, "expl", vals[0:-1,0:-1])
plt.figure()
plt.contourf(uSol_mor.X, uSol_mor.Y, uSol_mor.values, 40)
plt.colorbar()
plt.show()
plot(uSol);

"""for x in mesh.coordinates():
	if BD1.inside(x, True): print('%s ist links' % x)
	if BD2.inside(x, True): print('%s ist rechts unten' % x)
	if BN.inside(x, True): print('%s hat Neumann-0' % x)
	
# Print the Dirichlet conditions
print('Number of Dirichlet conditions:', len(bcs))
if V.ufl_element().degree() == 1:  # P1 elements
	d2v = dof_to_vertex_map(V)
	coor = mesh.coordinates()
	for i, bc in enumerate(bcs):
		print('Dirichlet condition %d' % i)
		boundary_values = bc.get_boundary_values()
		for dof in boundary_values:
			print('   dof %2d: u = %g' % (dof, boundary_values[dof]))
			if V.ufl_element().degree() == 1:
				print('    at point %s' %(str(tuple(coor[d2v[dof]].tolist()))))"""

print(uSol(60,40))
u = TrialFunction(V)
v = TestFunction(V)
a2 = kappa_fenics*dot(grad(u), grad(v))*dx
L2 = Constant(0)*v*dx
bcs2 = [DirichletBC(V, Constant(0.0), boundary_markers, 1), DirichletBC(V, Constant(0.0), boundary_markers, 2)]
A2, b2 = assemble_system(a2, L2, bcs2)
d = PointSource(V, Point(60, 40), 1)
d.apply(b2)
uSol2 = Function(V)
solve(A2, uSol2.vector(), b2)
vals = np.reshape(uSol2.compute_vertex_values(), (2**rect.resol+1, 2**rect.resol+1))
uSol2_mor = mor.mapOnRectangle(rect, "expl", vals[0:-1,0:-1])
plt.figure()
plt.contourf(uSol2_mor.X, uSol2_mor.Y, uSol2_mor.values, 40)
plt.colorbar()
plt.show()
plot(uSol2);
n = FacetNormal(mesh)
print(-2*assemble(kappa_fenics*dot(grad(uSol2), n)*ds(2)))

"""intfnc1 = Expression("pow((x[0]-40), 2) + pow((x[1]-20), 2) < 400 ? -1 : 0", degree=2)
int1 = intfnc1*uSol * dx
assemble(int1)
n = FacetNormal(mesh)
assemble(kappa_fenics*dot(grad(uSol2), n)*ds(2))"""
