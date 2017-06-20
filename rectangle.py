from __future__ import division
import numpy as np

class Rectangle:
	def __init__(self, p1=(0,0), p2=(1,1), resol=4):
		self.p1 = p1
		self.p2 = p2
		self.resol = resol
		self.x1 = p1[0]
		self.y1 = p1[1]
		self.x2 = p2[0]
		self.y2 = p2[1]
	
	def getXY(self): # return discretization of [x1,x2] and [y1,y2]
		return [np.linspace(self.x1, self.x2, 2**self.resol, endpoint=False), np.linspace(self.y1, self.y2, 2**self.resol, endpoint=False)]
	
	def getXYmeshgrid(self): # return domain discretization in form of np.meshgrid
		x, y = self.getXY()
		return np.meshgrid(x, y)
