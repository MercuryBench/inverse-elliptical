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

# Create mesh and define function space
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('0.01 + x[0]*x[0]', degree=2)
#u_D = Expression('0', degree=2)
# Define permeability
kappa = Expression('x[0] <= 0.5 ? 10 : 1', degree=2)

class myKappa(Expression):
	def eval(self, values, x):
		if x[0] <= 0.5:
			values[0] = 10
		elif x[0] >= 0.8:
			values[0] = 10
		else:
			values[0] = 1
m = ms.GaussianFourier2d(np.zeros((31,)), 3, 0.5).sample()

x = m.getX()
X, Y = np.meshgrid(x, x)

class myKappa2(Expression):
	def eval(self, values, x):
		values[0] = exp(m.handle(x))

#kappa = Expression('1', degree=2)
kappa = myKappa2(degree=2)
def boundary(x, on_boundary):
    return on_boundary

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()
ax.plot_wireframe(X, Y, m.values)
plt.show()

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = kappa*dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
interactive()
