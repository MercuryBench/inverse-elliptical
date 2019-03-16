from dolfin import *
from math import sin, cos
import sys
sys.path.append('..')
from rectangle import *
import mapOnRectangle as mor
from fwdProblem import *
import matplotlib.pyplot as plt


rect = Rectangle((0,0), (180,78), resol=7)

		
def boundary_D_boolean(x):
	if x[1] > 10**(-8):
		return True
	else:
		return False

u_D_mor = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)

fwd = linEllipt2dRectangle(rect, None, u_D_mor, boundary_D_boolean)
k0 = mor.mapOnRectangle(rect, "handle", lambda x,y: 0*x + 1)

uu = fwd.solveWithDiracRHS(k0, [-1, .75], [[80, 40], [100,20]])

plt.figure();
plt.ion()
plt.contourf(uu.values)
plt.show()

"""mesh = RectangleMesh(Point(rect.x1,rect.y1), Point(rect.x2,rect.y2), 2**rect.resol, 2**rect.resol)
def boundary_D_boolean(x):
	if x[1] > 10**(-8):
		return True
	else:
		return False


V = FunctionSpace(mesh, "P", 1)
mesh = RectangleMesh(Point(-1, -1), Point(1, 1), 2**rect.resol, 2**rect.resol)
def boundary_D(x, on_boundary):
			if on_boundary:
				if boundary_D_boolean(x):
					return True
				else:
					return False
			else:
				return False
				
boundary_D = boundary_D
u_D = morToFenicsConverter(u_D_mor, mesh, V)
bc = DirichletBC(V, u_D, boundary_D)

u = TrialFunction(V)
v = TestFunction(V)

k0 = mor.mapOnRectangle(rect, "handle", lambda x,y: 0*x + 1)
k = morToFenicsConverter(k0, mesh, V)

a = k*inner(grad(u), grad(v))*dx
L = Constant(0)*v*dx
bc = DirichletBC(V, Constant(0), DomainBoundary())
A, b = assemble_system(a, L, bc)

u = Function(V)
n_steps = 20

delta = PointSource(V, Point(80, 20), -1)
delta2 = PointSource(V, Point(100, 40), -.5)
delta3 = PointSource(V, Point(40, 30), -.25)
delta.apply(b)
delta2.apply(b)
delta3.apply(b)
solve(A, u.vector(), b)

plot(u, interactive=True)"""

