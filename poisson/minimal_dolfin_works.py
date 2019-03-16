from fenics import *
import numpy as np

mesh = UnitSquareMesh(1, 1)
a = MeshFunctionDouble(mesh, 2)
a[0] = 2.0
a[1] = 4.0
e = Expression("a", a=a, degree=0)
