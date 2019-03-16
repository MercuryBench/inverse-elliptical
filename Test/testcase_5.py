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

resol = 7
rect = Rectangle((0,0), (1,1), resol=resol)
rect2 = Rectangle((0,0), (1,1), resol=resol-2)
rect_coarse = Rectangle((0,0), (1,1), resol=resol-3)
gamma = 0.01

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
		return 4*(2*np.logical_and(y <= 0.3-0.1*x, True) + (-1)*np.logical_and(y>0.3-0.1*x, y<=0.8*x) + (-3)*np.logical_and(y > 0.3-0.1*x, np.logical_and(y > 0.8*x, y<=0.7-0.15*x)))
def u_D_term(x, y):
	return x*0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))
u_D2 = mor.mapOnRectangle(rect2, "handle", lambda x,y: u_D_term(x,y))
def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[1] <= tol:
			return True
		else:
			return False
#f = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)# (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)
f = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-0.25)**2 + (y-0.5)**2 < 0.05**2)*(-20.0))
f2 = mor.mapOnRectangle(rect2, "handle", lambda x, y: ((x-0.25)**2 + (y-0.5)**2 < 0.05**2)*(-20.0))

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
fwd2 = linEllipt2dRectangle(rect2, f2, u_D2, boundary_D_boolean)
m1 = GeneralizedGaussianWavelet2d(rect, 0.1, 2.0, resol)
m12 = GeneralizedGaussianWavelet2d(rect2, 0.1, 2.0, resol-2)
invProb = inverseProblem(fwd, m1, gamma)
invProb2 = inverseProblem(fwd2, m12, gamma)

if len(sys.argv) > 1:
	data = unpickleData(sys.argv[1])
	obspos = data["obspos"]
	obs = data["obs"]
	gamma = data["gamma"]
	resol = data["resol"]
	u0 = mor.mapOnRectangle(rect, "wavelet", data["u0_waveletcoeffs"])
	uOpt = mor.mapOnRectangle(rect, "wavelet", data["uOpt_waveletcoeffs"])
	uTruth_temp = mor.mapOnRectangle(rect_coarse, "handle", lambda x, y: myUTruth(x, y))
		
	uTruth = mor.mapOnRectangle(rect, "wavelet", uTruth_temp.waveletcoeffs)
	invProb.obs = obs
	invProb.obspos = obspos
	invProb.gamma = gamma
	invProb.resol = resol
	start = time.time()
	invProb.plotSolAndLogPermeability(uOpt)
	uOptList= [uOpt]
	for k in range(15):
		ucurrent = uOptList[-1]
		uOptNew = invProb.find_uMAP(ucurrent, nit=300, method='BFGS', adjoint=True);
		filename = "uMAP_adjoint" + str(k) + ".png";
		invProb.plotSolAndLogPermeability(uOptNew, save=filename)
		uOptList.append(uOptNew)
		# Create a text/plain message
		msg = MIMEText("Done!")

		# me == the sender's email address
		# you == the recipient's email address
		msg['Subject'] = 'Done'
		msg['From'] = "python"
		me = msg['From']
		msg['To'] = "phkwacker@gmail.com"
		you = msg['To']

		# Send the message via our own SMTP server, but don't include the
		# envelope header.
		s = smtplib.SMTP('localhost')
		s.sendmail(me, [you], msg.as_string())
		s.quit()
	end = time.time()
	print("Took " + str(end-start) + " seconds, which is " + str((end-start)/(3600.0)) + " hours.")
	data = {"obs": obs, "obspos": obspos, "resol": resol, "gamma": gamma, "u0_waveletcoeffs": u0.waveletcoeffs, "uOpt_waveletcoeffs": uOpt.waveletcoeffs}
	savefile = "data_adjoint_overnight.pkl"
	output = open('data_adjoint_overnight.pkl', 'wb')
	pickle.dump(data2, output)
	print("Saved as " + savefile)
else:
	N_obs = 500
	obspos = np.random.uniform(0, 1, (2, N_obs))
	obspos = [obspos[0,:], obspos[1, :]]
	invProb.obspos = obspos

	invProb2.obspos = obspos

	#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
	#uTruth = m1.sample()
	uTruth_temp = mor.mapOnRectangle(rect_coarse, "handle", lambda x, y: myUTruth(x, y))
		
	uTruth = mor.mapOnRectangle(rect, "wavelet", uTruth_temp.waveletcoeffs)

	obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
	invProb.plotSolAndLogPermeability(uTruth, obs=obs)
	invProb.obs = obs
	invProb2.obs = obs
	#u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	#uOpt = invProb.find_uMAP(u0, nit=3,  method='BFGS', adjoint=False)
	uOpt = u0
	D_primal = invProb.DI_adjoint_vec_wavelet(uOpt, version=2)
	D_dual = invProb.DI_adjoint_vec_wavelet(uOpt, version=0)
	D_dual2 = invProb2.DI_adjoint_vec_wavelet(uOpt, version=0)
	#uOpt = invProb.find_uMAP(uOpt, nit=20, method = 'BFGS', adjoint=True, version=0); invProb.plotSolAndLogPermeability(uOpt)
	plt.figure();
	plt.plot(D_primal, 'g')
	plt.plot(D_dual, 'r')


