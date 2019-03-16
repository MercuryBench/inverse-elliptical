import sys 
sys.path.append('..')
from haarWavelet2d import *


def unitwave(k):
	temp = np.zeros((16,))
	temp[k] = 1
	return mor.mapOnRectangle(Rectangle((0,0),(1,1),2), "wavelet", packWavelet(temp))

def plotMat(A, clim=(-30,30)):
	X = A.values
	fig, ax = plt.subplots()
	plt.ion()
	#imshow portion
	ax.imshow(X, interpolation='nearest', cmap="hot", clim=clim)
	#text portion
	diff = 1.
	min_val = 0.
	rows = X.shape[0]
	cols = X.shape[1]
	col_array = np.arange(min_val, cols, diff)
	row_array = np.arange(min_val, rows, diff)
	x, y = np.meshgrid(col_array, row_array)
	for col_val, row_val in zip(x.flatten(), y.flatten()):
	  c = X[row_val.astype(int), col_val.astype(int)]#'+' if X[row_val.astype(int),col_val.astype(int)] < 0.5 else '-' 
	  ax.text(col_val, row_val, c, va='center', ha='center')
	#set tick marks for grid
	ax.set_xticks(np.arange(min_val-diff/2, cols-diff/2))
	ax.set_yticks(np.arange(min_val-diff/2, rows-diff/2))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xlim(min_val-diff/2, cols-diff/2)
	ax.set_ylim(min_val-diff/2, rows-diff/2)
	ax.grid()
	plt.show()

wcu = np.array([-5,  2,  2, -5,  4, -1, -2,  4,  3,  1,  1, -5,  0, -3, -3, -2])

def extract(v, k):
	temp = np.zeros_like(v)
	temp[k] = v[k]
	return temp

A = mor.mapOnRectangle(Rectangle((0,0),(1,1),2), "wavelet", packWavelet(wcu))
plotMat(A, clim=(-40,15))

"""4:4 	8:3 	12:0 
5:-1 	9:1 	13:-3
6:-2 	10:1 	14:-3
7:4 	11:-5 15:-2"""
