from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, ceil, log10
import time
import sys 
sys.path.append('..')
from mpl_toolkits.mplot3d import Axes3D
from rectangle import *
import mapOnRectangle as mor
from haarWavelet2d import *

v = np.zeros((4,4))
v[0:2,0:2] = np.ones((2,2))
v[0,0] = 2


