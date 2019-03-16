from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from fenics import *

resol=8

mesh = RectangleMesh(Point(0,0), Point(1,1), 2**resol, 2**resol)
V = FunctionSpace(mesh, 'P', 1)

f = Expression("pow(x[0]-0.4, 2) + pow(x[1]-0.3, 2) <= 0.05 ? -10 : 0.0", degree=2)

boundary_markers = FacetFunction("size_t", mesh)

def boundaryclassifier(x):
	tol = 1E-14
	if near(x[0], 0, tol):
		return "X=0"
	elif near(x[1], 0, tol):
		return "Y=0"
	elif near(x[0], 1, tol):
		return "X=1"
	elif near(x[1], 1, tol):
		return "Y=1"
	else:
		return None

class BoundaryX0(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "X=0"
class BoundaryY0(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "Y=0"
class BoundaryX1(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "X=1"
class BoundaryY1(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "Y=1"


bx0 = BoundaryX0()
bx0.mark(boundary_markers, 1)
by0 = BoundaryY0()
by0.mark(boundary_markers, 2)
bx1 = BoundaryX1()
bx1.mark(boundary_markers, 3)
by1 = BoundaryY1()
by1.mark(boundary_markers, 4)

boundary_conditions = {1: {'Dirichlet': Constant(0.0)}, 2: {'Neumann':   Constant(0)}, 3: {'Dirichlet': Constant(1)}, 4: {'Dirichlet':   Expression("x[0]", degree=2)}}

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




boundary_conditions2 = {1: {'Dirichlet': Constant(0.0)}, 2: {'Neumann':   Constant(0.0)}, 3: {'Dirichlet': Constant(0.0)}, 4: {'Dirichlet':  Constant(0.0)}}
ds2 = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
bcs2 = []
for i in boundary_conditions2:
	if 'Dirichlet' in boundary_conditions2[i]:
		bc = DirichletBC(V, boundary_conditions2[i]['Dirichlet'], boundary_markers, i)
		bcs2.append(bc)
a2 = a
L2 = Constant(0)*v*dx 

A2, b2 = assemble_system(a2, L2, bcs2)
p1 = 0.6
p2 = 0.4
d = PointSource(V, Point(p1, p2), 1)
d.apply(b2)
uSol2 = Function(V)
solve(A2, uSol2.vector(), b2)
plot(uSol2)
n = FacetNormal(mesh)

print(uSol(p1,p2))
term1 = dot(grad(uSol2), n)*Expression("x[0]", degree=2)*ds2(1) + dot(grad(uSol2), n)*ds2(3) + dot(grad(uSol2), n)*Expression("x[0]", degree=2)*ds2(4)
print(assemble(f*uSol2*dx)-1*assemble(term1))

