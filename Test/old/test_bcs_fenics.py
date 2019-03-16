from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
import sys 
sys.path.append('..')
#from fwdProblem import *
from measures import *
import mapOnRectangle as mor
from rectangle import *



resol=4

mesh = RectangleMesh(Point(0,0), Point(1,1), 2**resol, 2**resol)
V = FunctionSpace(mesh, 'P', 1)

f = Constant(0)

boundary_markers = FacetFunction("size_t", mesh)


class BoundaryY0(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and near(x[1], 0, self.tol)
class BoundaryX1(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and near(x[0], 1, self.tol)
class BoundaryY1(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and near(x[1], 1, self.tol)

class BoundaryX0(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and near(x[0], 0, self.tol)


bx0 = BoundaryX0()
bx0.mark(boundary_markers, 4)
by0 = BoundaryY0()
by0.mark(boundary_markers, 1)
bx1 = BoundaryX1()
bx1.mark(boundary_markers, 2)
by1 = BoundaryY1()
by1.mark(boundary_markers, 3)

boundary_conditions = {4: {'Dirichlet': Constant(0.0)}, 1: {'Neumann':   Constant(0.0)}, 2: {'Dirichlet': Constant(1)}, 3: {'Dirichlet':   Expression("x[0]", degree=2)}}

bcs = []
for i in boundary_conditions:
	if 'Dirichlet' in boundary_conditions[i]:
		bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'], boundary_markers, i)
		bcs.append(bc)


u = TrialFunction(V)
v = TestFunction(V)


a = dot(grad(u), grad(v))*dx
L = f*v*dx # = 0


uSol = Function(V)
solve(a == L, uSol, bcs)
plot(uSol)
interactive()

# Print all vertices that belong to the boundary parts
for x in mesh.coordinates():
	if bx0.inside(x, True): print('%s is on x = 0' % x)
	if bx1.inside(x, True): print('%s is on x = 1' % x)
	if by0.inside(x, True): print('%s is on y = 0' % x)
	if by1.inside(x, True): print('%s is on y = 1' % x)
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
				print('    at point %s' %(str(tuple(coor[d2v[dof]].tolist()))))


vals = np.reshape(uSol.compute_vertex_values(), (2**rect.resol+1, 2**rect.resol+1))
uSol_mor = mor.mapOnRectangle(rect, "expl", vals[0:-1,0:-1])
plt.figure(); plt.ion()
plt.contourf(uSol_mor.values)
plt.colorbar()
plt.show()
