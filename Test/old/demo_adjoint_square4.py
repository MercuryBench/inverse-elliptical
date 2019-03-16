from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from fenics import *

resol=8

mesh = RectangleMesh(Point(0,0), Point(1,1), 2**resol, 2**resol)
V = FunctionSpace(mesh, 'P', 2)

f = Expression("pow(x[0]-0.4, 2) + pow(x[1]-0.3, 2) <= 0.05 ? -10 : 0.0", degree=5)

boundary_markers = FacetFunction("size_t", mesh)

def boundaryclassifier(x):
	tol = 1E-14
	if near(x[0], 0, tol):
		return "Left"
	elif near(x[1], 0, tol) or (near(x[0], 1, tol) and x[1] <= 0.5):
		return "RightBottom"
	elif near(x[1], 1, tol) or (near(x[0], 1, tol) and x[1] > 0.5):
		return "Neumann"
	else:
		return None
		
tol = 1E-14
indBL = Expression("x[0] < 1E-14 ? 1.0 : 0.0", degree=5)
indBRB = Expression("x[1] < 1E-14 || (x[0] > 1-1E-14 && x[1] <= 0.5+1E-14) ? 1.0 : 0.0", degree=5)
indBN = Expression("x[1] > 1 - 1E-14 || (x[0] > 1-1E-14 && x[1] > 0.5-1E-14) ? 1.0 : 0.0", degree=5)

class BoundaryLeft(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "Left"
class BoundaryRB(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "RightBottom"
class BoundaryN(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "Neumann"


bl = BoundaryLeft()
bl.mark(boundary_markers, 1)
brb = BoundaryRB()
brb.mark(boundary_markers, 2)
bn = BoundaryN()
bn.mark(boundary_markers, 3)

boundary_conditions = {1: {'Dirichlet': Constant(0.0)}, 2: {'Dirichlet':   Expression("x[0]+x[1]", degree=5)}, 3: {'Neumann': Constant(0.0)}}

bcs = []
for i in boundary_conditions:
	if 'Dirichlet' in boundary_conditions[i]:
		bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'], boundary_markers, i)
		bcs.append(bc)


u = TrialFunction(V)
v = TestFunction(V)


a = dot(grad(u), grad(v))*dx
L = f*v*dx 


uSol = Function(V)
solve(a == L, uSol, bcs)
plot(uSol); #interactive()




boundary_conditions2 = {1: {'Dirichlet': Constant(0.0)}, 2: {'Dirichlet':   Constant(0.0)}, 3: {'Neumann': Constant(0.0)}}
ds2 = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
ds = Measure("ds", domain=mesh)
bcs2 = []
for i in boundary_conditions2:
	if 'Dirichlet' in boundary_conditions2[i]:
		bc = DirichletBC(V, boundary_conditions2[i]['Dirichlet'], boundary_markers, i)
		bcs2.append(bc)
a2 = a
l = Constant(0)
L2 = l*v*dx

A2, b2 = assemble_system(a2, L2, bcs2)
d = PointSource(V, Point(0.6, 0.4), 1)
d.apply(b2)

uSol2 = Function(V)
solve(A2, uSol2.vector(), b2)
plot(uSol2)
n = FacetNormal(mesh)




"""for x in mesh.coordinates():
	if bl.inside(x, True): print('%s is left' % x)
	if brb.inside(x, True): print('%s is right/bottom' % x)
	if bn.inside(x, True): print('%s is Neumann' % x)
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

primal = uSol(0.6,0.4)

t1 = assemble(uSol*dot(grad(uSol2), n)*ds)
t2 = assemble(dot(grad(uSol),grad(uSol2))*dx)
step1 = t2 - t1

t3 = assemble(uSol*dot(grad(uSol2), n)*ds)
t3var = assemble(uSol*dot(grad(uSol2), n)*ds2(1) + uSol*dot(grad(uSol2), n)*ds2(2) + uSol*dot(grad(uSol2), n)*ds2(3))



t4 = assemble(uSol2*dot(grad(uSol),n)*ds)
t5 = assemble(f*uSol2*dx)

step2 = t5 + t4 - t3

dual = -assemble(dot(grad(uSol2), n)*boundary_conditions[2]['Dirichlet']*indBRB*ds) + assemble(f*uSol2*dx)
#term1 = dot(grad(uSol2), n)*Expression("x[0]+x[1]", degree=2)*ds2(2)
print(primal)
print(dual)
