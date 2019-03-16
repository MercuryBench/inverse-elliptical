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

resol = 3
rect = Rectangle((0,0), (1,1), resol=resol)
rect_accurate = Rectangle((0,0), (1,1), resol=resol+3)
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
		#return  -4/log10(e)*np.logical_or(np.logical_and(np.logical_and(x < 0.5, x >= 0.4375), y < 0.5), np.logical_and(np.logical_and( x< 0.5 , x >= 0.4375) ,y >= 0.625 - tol)) + 2/log10(e)*np.logical_and(np.logical_and(x < 0.8125, x >= 0.75), np.logical_and(y >= 0.1875, y < 0.75)) + 0
		return np.logical_or(np.logical_and(np.logical_and((x+0.5)**2 + (y-0.2)**2 <= 1.5, (x+0.5)**2 + (y-0.25)**2 >= 1), y <=0.3), np.logical_and(np.logical_and((x+0.5)**2 + (y-0.2)**2 <= 1.5, (x+0.5)**2 + (y-0.25)**2 >= 1), y >=0.5))*(-40.0)
		#return x*0+1

def u_D_term(x, y):
	return np.logical_and(x >= 0.5, y <= 0.625)*0.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))
u_D_accurate = mor.mapOnRectangle(rect_accurate, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[1] <= tol:
			return True
		else:
			return False
f = mor.mapOnRectangle(rect, "handle", lambda x, y: ( ((x-.2)**2 + (y-.75)**2) < 0.1**2)*(-200.0)) #+ (((x-.8)**2 + (y-.75)**2) < 0.05**2)*20.0)
f_accurate = mor.mapOnRectangle(rect_accurate, "handle", lambda x, y: ( ((x-.2)**2 + (y-.75)**2) < 0.1**2)*(-200.0))

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
fwd_accurate = linEllipt2dRectangle(rect_accurate, f_accurate, u_D_accurate, boundary_D_boolean)
m1 = GeneralizedGaussianWavelet2d(rect, 0.01, 1.5, resol)
m1_accurate = GeneralizedGaussianWavelet2d(rect_accurate, 0.01, 1.5, resol+3)
invProb = inverseProblem(fwd, m1, gamma)
invProb_accurate = inverseProblem(fwd_accurate, m1_accurate, gamma)


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
	start = time.time()
	for k in range(15):
		ucurrent = uOptList[-1]
		uOptNew = invProb.find_uMAP(ucurrent, nit=300, method='BFGS', adjoint=True);
		filename = "uMAP4_adjoint" + str(k) + ".png";
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
	data2 = {"obs": obs, "obspos": obspos, "resol": resol, "gamma": gamma, "u0_waveletcoeffs": u0.waveletcoeffs, "uOpt_waveletcoeffs": uOptList[-1].waveletcoeffs}
	savefile = "data_temp.pkl"
	output = open(savefile, 'wb')
	pickle.dump(data2, output)
	print("Saved as " + savefile)
else:
	N_obs = 500
	obspos = np.random.uniform(0, 1, (2, N_obs))
	obspos = [obspos[0,:], obspos[1, :]]
	invProb.obspos = obspos
	invProb_accurate.obspos = obspos
	#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
	#uTruth = m1.sample()
	uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth(x, y))

	obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
	invProb.plotSolAndLogPermeability(uTruth, obs=obs)
	invProb.obs = obs
	invProb_accurate.obs = obs
	#u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	#uOpt = invProb.find_uMAP(u0, nit=3,  method='BFGS', adjoint=False)
	#start = time.time()
	uOpt = u0
	#uOpt = invProb.find_uMAP(uOpt, nit=50, method = 'BFGS', adjoint=True, version=0);invProb.plotSolAndLogPermeability(uOpt)
	#uOpt2 = invProb.find_uMAP(uTruth, nit=50, method = 'BFGS', adjoint=True, version=0);invProb.plotSolAndLogPermeability(uOpt2)
	u = uTruth
	
	DPhi_pr = invProb.DPhi_vec_wavelet(u)
	DPhi_ad = invProb.DPhi_adjoint_vec_wavelet(u)
	DPhi_ad2 = invProb.DPhi_adjoint_vec_wavelet(u, version=0)
	#DPhi_ad2_accurate = invProb_accurate.DPhi_adjoint_vec_wavelet(u, version=0)
	print(DPhi_pr[0:16])
	print(DPhi_ad[0:16])
	print((DPhi_pr[0:16]-DPhi_ad[0:16])/DPhi_pr[0:16])
	print((DPhi_pr[100:116]-DPhi_ad[100:116])/DPhi_pr[100:116])
	plt.figure(); plt.plot(DPhi_pr[0:16]); plt.plot(DPhi_ad[0:16], 'r')
	plt.figure(); plt.plot(DPhi_pr[100:116]); plt.plot(DPhi_ad[100:116], 'r')
	Fu_, Fu = invProb.Ffnc(u, pureFenicsOutput="both")
	mesh = invProb.fwd.mesh
	V = invProb.fwd.V
	kappa = mor.mapOnRectangle(rect, "handle", lambda x,y: np.exp(u.handle(x,y)))
	
	
	discrepancy = obs - Fu.handle(obspos[0], obspos[1])
	weights = -discrepancy/gamma**2
	wtildeSol = invProb.fwd.solveWithDiracRHS(kappa, weights, zip(obspos[0][:], obspos[1][:]), pureFenicsOutput=True)
	k = morToFenicsConverterHigherOrder(kappa, mesh, V)
	fnc = project(k*dot(grad(Fu_), grad(wtildeSol)), V)
	
	"""uOptList = [uOpt]
	for k in range(1, 24):
	ucurrent = uOptList[-1]
	uOptNew = invProb.find_uMAP(ucurrent, nit=100, method='BFGS', adjoint=True, version=0);
	filename = "uMAP4_quickadjoint_" + str(k) + ".png";
	invProb.plotSolAndLogPermeability(uOptNew, save=filename)
	uOptList.append(uOptNew)
	# Create a text/plain message"""
	"""msg = MIMEText("Done!")

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
	s.quit()"""
	"""end = time.time()
	print("Took " + str(end-start) + " seconds, which is " + str((end-start)/(3600.0)) + " hours.")
	data2 = {"obs": obs, "obspos": obspos, "resol": resol, "gamma": gamma, "u0_waveletcoeffs": u0.waveletcoeffs, "uOpt_waveletcoeffs": uOptList[-1].waveletcoeffs}
	savefile = "testcase_4_tempname.pkl"
	output = open(savefile, 'wb')
	pickle.dump(data2, output)
	print("Saved as " + savefile)"""


