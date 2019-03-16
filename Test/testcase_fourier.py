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
gamma = 0.01


def u_D_term(x, y):
	return x*0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[1] <= tol:
			return True
		else:
			return False

#f = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)# (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)
#f = Constant(0)
f = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-0.4)**2 + (y-0.7)**2 < 0.1**2)*(-20.0))

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
#m1 = GeneralizedGaussianWavelet2d(rect, 0.1, 1.5, resol)
m = GaussianFourier2d(rect, np.zeros((11,11)), 2.0, 1.0)
invProb = inverseProblem(fwd, m, gamma)


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
	"""uOptList= [uOpt]
	start = time.time()
	for k in range(15):
		ucurrent = uOptList[-1]
		uOptNew = invProb.find_uMAP(ucurrent, nit=300, method='BFGS', adjoint=True, version=0);
		filename = "uMAP_3_adjoint" + str(k) + ".png";
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
	savefile = "data_test3.pkl"
	output = open(savefile, 'wb')
	pickle.dump(data2, output)
	print("Saved as " + savefile)"""
else:
	N_obs = 500
	
	obsposx = np.random.uniform(0, 1, (N_obs,))
	obsposy = np.random.uniform(0, 1, (N_obs,))
	obspos = [obsposx, obsposy]
	invProb.obspos = obspos

	#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
	uTruth = m.sample()
	#uTruth1 = mor.mapOnRectangle(rectTruth, "handle", lambda x, y: myUTruth(x, y))
	#uTruth2 = mor.mapOnRectangle(rectTruth, "fourier", np.copy(uTruth1.fouriermodes))

	obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
	invProb.plotSolAndLogPermeability(uTruth, obs=obs)
	invProb.obs = obs
	#u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	m2mean = m._mean 
	u02 = mor.mapOnRectangle(rect, "fourier", m2mean)
	#uOpt = invProb.find_uMAP(u0, nit=3,  method='BFGS', adjoint=False)
	uOpt = u02
	invProb.plotSolAndLogPermeability(uOpt)
	#uOpt2 = invProb2.find_uMAP(uOpt2, nit=10, method = 'BFGS', adjoint=True, version=0);invProb2.plotSolAndLogPermeability(uOpt2)
	DD = invProb.DPhi_vec_fourier(uOpt, obs)
	DD2 = invProb.DPhi_adjoint_vec_fourier(uOpt)
"""start = time.time()
uOpt = invProb.find_uMAP(u0, nit=10, method = 'BFGS', adjoint=True, version=0)
invProb.plotSolAndLogPermeability(uOpt, save="uMAP3_quickadjoint_0.png")
print("Done")
uOptList = [uOpt]
for k in range(1, 24):
ucurrent = uOptList[-1]
uOptNew = invProb.find_uMAP(ucurrent, nit=100, method='BFGS', adjoint=True, version=0);
filename = "uMAP3_quickadjoint_" + str(k) + ".png";
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
savefile = "testcase_3_2_tempname.pkl"
output = open(savefile, 'wb')
pickle.dump(data2, output)
print("Saved as " + savefile)"""



