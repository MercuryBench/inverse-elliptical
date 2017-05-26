"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary
  u_D = 1 + x^2 + 2y^2
    f = -6
"""


from __future__ import print_function
from fenics import *
import measures as ms
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

tol = 1E-14

# Create mesh and define function space
mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition

u_D = Expression('(x[0] >= 0.5 && x[1] <= 0.6) ? 1 : 0', degree=2)

# Define permeability
#kappa = Expression('x[0] <= 0.5 ? 10 : 1', degree=2) # simple two-stripe permeability

class myKappa(Expression):
	def eval(self, values, x):
		if x[0] <= 0.5+tol:
			values[0] = 10
		elif x[0] >= 0.8-tol:
			values[0] = 10
		else:
			values[0] = 1
			
class myKappaTestbed(Expression): # more complicated topology
	def eval(self, values, x):
		if x[0] <= 0.5 +tol  and x[0] >= 0.45 - tol and x[1] <= 0.5+tol:
			values[0] = 0.0001
		elif x[0] <= 0.5+tol and x[0] >= 0.45 - tol and x[1] >= 0.6 - tol:
			values[0] = 0.0001
		elif x[0] <= 0.75 + tol and x[0] >= 0.7 - tol and x[1] >= 0.2 - tol and x[1] <= 0.8+tol:
			values[0] = 100
		else:
			values[0] = 1

class fTestbed(Expression): # more complicated source and sink term
	def eval(self, values, x):
		if pow(x[0]-0.6, 2) + pow(x[1]-0.85, 2) <= 0.1*0.1:
			values[0] = -20
		elif pow(x[0]-0.2, 2) + pow(x[1]-0.75, 2) <= 0.1*0.1:
			values[0] = 20
		else:
			values[0] = 0

# sample log permeability
m = ms.GaussianFourier2d(np.zeros((21,21)), 2., 2).sample()
m1 = ms.GeneralizedGaussianWavelet2d(1, 1.0, 8).sample()

# coords of mesh vertices
coords = mesh.coordinates().T

# evaluate permeability in vertices
vals = np.exp(m1.handle(coords))

kappa = Function(V)


# fill permeability function with values using Miguel's dof juggling
kappa.vector().set_local(vals[dof_to_vertex_map(V)])

# alternative: brute force Expression (not recommended)
class myKappaGaussianFourier(Expression):
	def eval(self, values, x):
		values[0] = exp(m.handle(x))
class myKappaGaussianWavelet2d(Expression):
	def eval(self, values, x):
		values[0] = exp(m1.handle(x))


#kappa = Expression('1', degree=2)
#kappa = myKappaTestbed(degree=2) #for debugging/ground truth purposes
#kappa = myKappaGaussianFourier(degree = 2) # non-recommended old version
#kappa = myKappaGaussianWavelet2d(degree=2)
def boundaryD(x, on_boundary): # special Dirichlet boundary condition
	if on_boundary:
		if x[0] >= 0.6-tol and x[1] <= 0.5:
			return True
		elif x[0] <= tol: # obsolete
			return True
		else:
			return False
	else:
		return False


def boundary(x, on_boundary):
	return on_boundary

# plot permeability
x = m1.getX()
X, Y = np.meshgrid(x, x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()
ax.plot_wireframe(X, Y, np.exp(m1.values))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()
ax.plot_wireframe(X, Y, np.exp(m.values))
plt.show()


bc = DirichletBC(V, u_D, boundaryD)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = fTestbed(degree = 2)
#f = Expression('100*exp(-pow(x[0]-0.2, 2) - pow(x[1]-0.75,2))', degree=2)
L = f*v*dx

a = kappa*dot(grad(u), grad(v))*dx
# Compute solution
uSol = Function(V)
solve(a == L, uSol, bc)

# Plot solution and mesh
plot(uSol)
plot(mesh)
#plot(kappa)

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << uSol

# Compute error in L2 norm
error_L2 = errornorm(u_D, uSol, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = uSol.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
interactive()
