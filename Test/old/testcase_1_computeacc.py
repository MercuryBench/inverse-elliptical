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

resol = 4
rect = Rectangle((0,0), (1,1), resol=resol)
rect_fine1 = Rectangle((0,0), (1,1), resol=resol+1)
rect_fine2 = Rectangle((0,0), (1,1), resol=resol+2)
rect_fine3 = Rectangle((0,0), (1,1), resol=resol+3)
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
		return  -4/log10(e)*np.logical_or(np.logical_and(np.logical_and(x < 0.5, x >= 0.25), y < 0.25), np.logical_and(np.logical_and( x< 0.5 , x >= 0.25) ,y >= 0.375)) + 0
def u_D_term(x, y):
	return np.logical_and(x >= 1-10**-8, True)*2.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] >= 1-10**-8:
			return True
		elif x[0] <= 10**-8:
			return True
		else:
			return False

#f = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)# (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)
f = Constant(0)

fwd0 = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
fwd1 = linEllipt2dRectangle(rect_fine1, f, u_D, boundary_D_boolean)
fwd2 = linEllipt2dRectangle(rect_fine2, f, u_D, boundary_D_boolean)
fwd3 = linEllipt2dRectangle(rect_fine3, f, u_D, boundary_D_boolean)
m1 = GeneralizedGaussianWavelet2d(rect, 0.1, 1.5, resol)
invProb0 = inverseProblem(fwd0, m1, gamma)
invProb1 = inverseProblem(fwd1, m1, gamma)
invProb2 = inverseProblem(fwd2, m1, gamma)
invProb3 = inverseProblem(fwd3, m1, gamma)

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
	uOptList= [uOpt]
	for k in range(15):
		ucurrent = uOptList[-1]
		uOptNew = invProb.find_uMAP(ucurrent, nit=50, method='BFGS', adjoint=True);
		filename = "uMAP_adjoint" + str(k) + ".png";
		invProb.plotSolAndLogPermeability(uOptNew)#, save=filename)
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
	data2 = {"obs": obs, "obspos": obspos, "resol": resol, "gamma": gamma, "u0_waveletcoeffs": u0.waveletcoeffs, "uOpt_waveletcoeffs": uOptList[-1].waveletcoeffs}
	savefile = "data_adjoint_overnight.pkl"
	output = open('data_adjoint_overnight.pkl', 'wb')
	pickle.dump(data2, output)
	print("Saved as " + savefile)
else:
	N_obs = 10
	obspos1 = np.random.uniform(0, 1, (2, N_obs))
	
	obspos = [obspos1[0,:], obspos1[1,:]]
	invProb0.obspos = obspos
	invProb1.obspos = obspos
	invProb2.obspos = obspos
	invProb3.obspos = obspos

	#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
	#uTruth = m1.sample()
	uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth(x, y))
	uTruth0 = mor.mapOnRectangle(rect, "wavelet", uTruth.waveletcoeffs)
	uTruth1 = mor.mapOnRectangle(rect_fine1, "wavelet", uTruth.waveletcoeffs)
	uTruth2 = mor.mapOnRectangle(rect_fine2, "wavelet", uTruth.waveletcoeffs)
	uTruth3 = mor.mapOnRectangle(rect_fine3, "wavelet", uTruth.waveletcoeffs)

	obs0 = invProb0.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
	obs3 = invProb3.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
	invProb0.obs = obs0
	invProb3.obs = obs3
	invProb0.plotSolAndLogPermeability(uTruth0, obs=obs0)
	invProb3.plotSolAndLogPermeability(uTruth3, obs=obs3)
	#u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	u00 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	u03 = mor.mapOnRectangle(rect_fine3, "wavelet", m1._mean)
	#uOpt = invProb.find_uMAP(u0, nit=3,  method='BFGS', adjoint=False)
	uOpt0 = u00
	uOpt3 = u03
	uOpt0 = invProb0.find_uMAP(uOpt0, nit=50, method = 'BFGS', adjoint=True, version=0);invProb0.plotSolAndLogPermeability(uOpt0)
	uOpt3 = invProb3.find_uMAP(uOpt3, nit=50, method = 'BFGS', adjoint=True, version=0);invProb3.plotSolAndLogPermeability(uOpt3)


