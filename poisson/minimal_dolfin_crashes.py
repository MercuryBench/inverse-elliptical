from fenics import *
import numpy as np

mesh = UnitSquareMesh(1, 1)
a = MeshFunctionDouble(mesh, 0)
a.set_values(np.array([1.0,2.0,3.0,4.0]))
e = Expression("a", a=a, degree=0)
