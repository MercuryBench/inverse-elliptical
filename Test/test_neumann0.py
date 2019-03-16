from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from fenics import *

resol=3

mesh = RectangleMesh(Point(0,0), Point(1,1), 2**resol, 2**resol)
V = FunctionSpace(mesh, 'CG', 2)

f = Expression("pow(x[0]-0.4, 2) + pow(x[1]-0.3, 2) <= 0.05 ? -10 : 0.0", degree=5)

boundary_markers = FacetFunction("size_t", mesh)

def boundaryclassifier(x):
	tol = 1E-14
	if near(x[0], 0, tol):
		return "DirichletLeft"
	elif near(x[1], 0, tol) or (near(x[0], 1, tol) and x[1] <= 0.5):
		return "DirichletRightBottom"
	elif near(x[1], 1, tol) or (near(x[0], 1, tol) and x[1] > 0.5):
		return "NeumannRightTop"
	else:
		return None
		

class BoundaryLeft(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "DirichletLeft"
class BoundaryRB(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "DirichletRightBottom"
class BoundaryN(SubDomain):
	tol = 1E-14
	def inside(self, x, on_boundary):
		return on_boundary and boundaryclassifier(x) == "NeumannRightTop"


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
plot(uSol)


boundary_conditions2 = {1: {'Dirichlet': Constant(0.0)}, 2: {'Dirichlet':   Constant(0.0)}, 3: {'Neumann': Constant(0.0)}}
ds_surf = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
n = FacetNormal(mesh)

print(assemble(uSol*ds_surf(1)))
print(assemble(dot(grad(uSol),n)*ds_surf(3)))
