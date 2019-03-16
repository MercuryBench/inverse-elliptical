from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
import math
import sys 
sys.path.append('..')
import mapOnRectangle as mor
from fwdProblem import *
from invProblem2d import *
from rectangle import *
import smtplib
from email.mime.text import MIMEText

resol = 6
rect = Rectangle((0,0), (1,1), resol=resol)
rect_coarse = Rectangle((0,0), (1,1), resol=resol-2)
gamma = 0.0005

def myUTruth(x, y):
		"""if x[0] <= 0.5 +tol  and x[0] >= 0.45 - tol and x[1] <= 0.5+tol:
			return -4
		elif x[0] <= 0.5+tol and x[0] >= 0.45 - tol and x[1] >= 0.6 - tol:
			return -4
		elif x[0] <= 0.75 + tol and x[0] >= 0.7 - tol and x[1] >= 0.2 - tol and x[1] <= 0.8+tol:
			return 2
		else:
			return 0"""
		#if x.ndim == 1 and x.shape[0] == 2:
		return (-2)*np.logical_and(y <= 0.3-0.1*x, True) + (-1)*np.logical_and(y>0.3-0.1*x, y<=0.8*x) + (2)*np.logical_and(y > 0.3-0.1*x, np.logical_and(y > 0.8*x, y<=0.7-0.15*x))
def u_D_term(x, y):
	return x*0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[1] <= tol:
			return True
		else:
			return False
#f = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)# (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)
f = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-0.7)**2 + (y-0.8)**2 < 0.05**2)*(-20.0))

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
m1 = GeneralizedGaussianWavelet2d(rect, 0.1, 1.5, resol)
invProb = inverseProblem(fwd, m1, gamma)

if len(sys.argv) > 1:
	data = unpickleData(sys.argv[1])
	obspos = data["obspos"]
	obs = data["obs"]
	gamma = data["gamma"]
	resol = data["resol"]
	u0 = mor.mapOnRectangle(rect, "wavelet", data["u0_waveletcoeffs"])
	uOpt = mor.mapOnRectangle(rect, "wavelet", data["uOpt_waveletcoeffs"])
	uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth(x, y))
	invProb.obs = obs
	invProb.obspos = obspos
	invProb.gamma = gamma
	invProb.resol = resol
	start = time.time()
	invProb.plotSolAndLogPermeability(uOpt)
else:
	N_obs = 150
	obspos =[np.array([ 0.71820781,  0.61536396,  0.21460308,  0.41379042,  0.71508557,
        0.99584303,  0.47844673,  0.23251669,  0.99222769,  0.53760169,
        0.54020305,  0.87616242,  0.88526929,  0.12942788,  0.7317283 ,
        0.14600608,  0.93271015,  0.68590646,  0.00898568,  0.41810677,
        0.42155167,  0.9348932 ,  0.27673156,  0.45667992,  0.40223538,
        0.48239475,  0.9629205 ,  0.5772067 ,  0.89768373,  0.12809556,
        0.77506126,  0.91918543,  0.0169915 ,  0.68849373,  0.88097219,
        0.10091014,  0.6131689 ,  0.04794616,  0.64793026,  0.8691139 ,
        0.17524361,  0.98903011,  0.09985323,  0.65921084,  0.01243067,
        0.44348648,  0.73558419,  0.67832169,  0.12170886,  0.96110047,
        0.52276505,  0.80598881,  0.88006273,  0.88370167,  0.99806603,
        0.03815272,  0.86241194,  0.05895183,  0.21658541,  0.315514  ,
        0.80090139,  0.06398846,  0.90016533,  0.10781413,  0.68450314,
        0.85623455,  0.26795053,  0.12317224,  0.4887334 ,  0.81370774,
        0.91155435,  0.87137815,  0.012642  ,  0.06169672,  0.46105482,
        0.08242722,  0.67484909,  0.48472863,  0.47958067,  0.45477776,
        0.87746415,  0.58648917,  0.00912683,  0.39829417,  0.58792888,
        0.91530248,  0.37551304,  0.55180134,  0.68680397,  0.5040464 ,
        0.46922942,  0.51799969,  0.87473308,  0.19604167,  0.3638077 ,
        0.74716313,  0.98009203,  0.39394891,  0.02764732,  0.59541678,
        0.13350073,  0.86437121,  0.42254423,  0.65008462,  0.39947893,
        0.07704183,  0.74199788,  0.48719646,  0.41805548,  0.76238931,
        0.50764875,  0.36261733,  0.65129952,  0.9956839 ,  0.01321819,
        0.24351804,  0.04929066,  0.77033606,  0.41386653,  0.08595658,
        0.13632861,  0.78903774,  0.14642196,  0.69383688,  0.65842389,
        0.34320255,  0.52413468,  0.31959195,  0.37765343,  0.40309586,
        0.48596773,  0.50501601,  0.14006812,  0.61978371,  0.83843071,
        0.06755591,  0.77373772,  0.08461378,  0.14526229,  0.57368769,
        0.81330166,  0.85425683,  0.11463711,  0.94958498,  0.6740511 ,
        0.26682618,  0.86980714,  0.34692778,  0.79437007,  0.78457971]), np.array([ 0.93981941,  0.93393646,  0.34080725,  0.84597362,  0.28025561,
        0.26988527,  0.06764752,  0.74393463,  0.36281563,  0.28610681,
        0.40588855,  0.79851176,  0.49252364,  0.34712235,  0.30699621,
        0.74851437,  0.75284011,  0.30773176,  0.40928426,  0.27367561,
        0.42387282,  0.31057594,  0.84629414,  0.81299803,  0.64862138,
        0.14048374,  0.73359715,  0.0814939 ,  0.4318018 ,  0.04810533,
        0.1848511 ,  0.36425873,  0.37923228,  0.23806547,  0.39889903,
        0.07242274,  0.04775527,  0.36404654,  0.2749374 ,  0.75016979,
        0.57579581,  0.06507159,  0.15000598,  0.05970805,  0.28105538,
        0.51609075,  0.77210313,  0.12159067,  0.30673821,  0.08388259,
        0.50517141,  0.33602999,  0.82849047,  0.17089808,  0.14717778,
        0.80264785,  0.25710686,  0.63106562,  0.43976658,  0.41041194,
        0.84116544,  0.65324944,  0.20007899,  0.97109744,  0.86042668,
        0.4326131 ,  0.09060329,  0.70512814,  0.08747801,  0.79777019,
        0.56525443,  0.38145327,  0.41456539,  0.78913698,  0.88389736,
        0.88708395,  0.39364302,  0.3274575 ,  0.23940161,  0.57185253,
        0.58069508,  0.12586795,  0.02797865,  0.72805366,  0.19574313,
        0.69082712,  0.37737158,  0.32018761,  0.71929984,  0.72326576,
        0.37089072,  0.63954305,  0.97495632,  0.2754366 ,  0.61042491,
        0.41856409,  0.29512827,  0.51260299,  0.88810313,  0.39861314,
        0.75490779,  0.84059845,  0.64568309,  0.00640482,  0.6771873 ,
        0.14302325,  0.53098199,  0.40706304,  0.28392694,  0.5424342 ,
        0.02788521,  0.65283813,  0.60877472,  0.19305001,  0.66468995,
        0.18768019,  0.30256114,  0.60140598,  0.52143598,  0.77877203,
        0.944851  ,  0.59357812,  0.05837093,  0.03484918,  0.01380712,
        0.82362101,  0.35279256,  0.41272375,  0.45824288,  0.71529376,
        0.44210919,  0.59234605,  0.76178363,  0.19891549,  0.13299069,
        0.98515161,  0.39967782,  0.47834306,  0.33296685,  0.30782319,
        0.01907857,  0.34568011,  0.06585478,  0.232502  ,  0.80427097,
        0.30080206,  0.00710588,  0.31702161,  0.28798715,  0.32257366])]
	#obspos = np.random.uniform(0, 1, (2, N_obs))
	#obspos = [obspos[0,:], obspos[1, :]]
	invProb.obspos = obspos

	#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
	#uTruth = m1.sample()
	uTruth = mor.mapOnRectangle(rect_coarse, "handle", lambda x, y: myUTruth(x, y))
	uTruth = mor.mapOnRectangle(rect, "wavelet", uTruth.waveletcoeffs)

	obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
	invProb.plotSolAndLogPermeability(uTruth, obs=obs)
	invProb.obs = obs
	#u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	#uOpt = invProb.find_uMAP(u0, nit=3,  method='BFGS', adjoint=False)
	uOpt = u0
	uOpt = invProb.find_uMAP(uOpt, nit=50, method = 'BFGS', adjoint=True, version=0); invProb.plotSolAndLogPermeability(uOpt)


