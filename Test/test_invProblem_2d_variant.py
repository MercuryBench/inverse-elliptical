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

rect = Rectangle((0,0), (1,1), resol=7)
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
		return  -4/log10(e)*np.logical_or(np.logical_and(np.logical_and(x <= 0.5 +tol, x >= 0.45 - tol), y <= 0.5+tol), np.logical_and(np.logical_and( x<= 0.5+tol , x >= 0.45 - tol) ,y >= 0.6 - tol)) + 2/log10(e)*np.logical_and(np.logical_and(x <= 0.75 + tol, x >= 0.7 - tol), np.logical_and(y >= 0.2 - tol, y <= 0.8+tol)) + 0
def myUTruth3(x,y):
	return 1 - 4.0*np.logical_and(np.logical_and(x >= 0.375, x < 0.75), y < 0.625)
def myUTruth2(x,y):
	return 1.0 - 2.0*np.logical_or(np.logical_and(np.logical_and(x>=0.25, x<0.75), y < 0.5), np.logical_and(np.logical_and(x>=0.25, x<0.75), y >= 0.75))
def myUTruth4(x,y):
	return 1.0 - 2.0*np.logical_and(x>=0.5, y>=0)
def u_D_term(x, y):
	return np.logical_and(x >= 0.5, y <= 0.6)*2.0

u_D = mor.mapOnRectangle(rect, "handle", lambda x,y: u_D_term(x,y))

def boundary_D_boolean(x): # special Dirichlet boundary condition
		if x[0] >= 0.6-tol and x[1] <= 0.5:
			return True
		elif x[0] <= 10**-8:
			return True
		else:
			return False

f = mor.mapOnRectangle(rect, "handle", lambda x, y: 0*x)# (((x-.6)**2 + (y-.85)**2) < 0.1**2)*(-20.0) + (((x-.2)**2 + (y-.75)**2) < 0.1**2)*20.0)

fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)
m1 = GeneralizedGaussianWavelet2d(rect, 0.1, 1.5, 7)
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
	for k in range(12, 13):
		uOpt = invProb.find_uMAP(uOpt, nit=30, method='BFGS');filename = "uMAP" + str(k) + ".png";invProb.plotSolAndLogPermeability(uOpt, save=filename)
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
	
	
	
else:
	

	N_obs = 700
	obspos = np.random.uniform(0, 1, (2, N_obs))
	obspos = [obspos[0,:], obspos[1, :]]
	invProb.obspos = obspos

	#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
	#uTruth = m1.sample()
	uTruth = mor.mapOnRectangle(rect, "handle", lambda x, y: myUTruth(x, y))
	
	obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
	invProb.plotSolAndLogPermeability(uTruth, obs=obs, save="ground_truth.png")
	invProb.obs = obs
	u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean)
	#uOpt = invProb.find_uMAP(u0, nit=100, nfev=100)
	start = time.time()
	uOpt = u0
	for k in range(12):
		uOpt = invProb.find_uMAP(uOpt, nit=30, method='BFGS')
		filename = "uMAP" + str(k) + ".png"
		invProb.plotSolAndLogPermeability(uOpt, save=filename)
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

# Import the email modules we'll need

# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.


#u_new, u_new_mean, us = invProb.EnKF(obs, 128, KL=False, N = 1)
#invProb.plotSolAndLogPermeability(u_new_mean)






print("ground truth:\tI = " + str(invProb.I(uTruth)))
print("\t\tPhi = " + str(invProb.Phi(uTruth)))
print("\t\tnorm = " + str(invProb.prior.normpart(uTruth)))
print("uMAP:\t\tI = " + str(invProb.I(uOpt)))
print("\t\tPhi = " + str(invProb.Phi(uOpt)))
print("\t\tnorm = " + str(invProb.prior.normpart(uOpt)))
data = {"obs": obs, "obspos": obspos, "resol": resol, "gamma": gamma, "u0_waveletcoeffs": u0.waveletcoeffs, "uOpt_waveletcoeffs": uOpt.waveletcoeffs}
output = open('data_longrun_+2.pkl', 'wb')
pickle.dump(data, output)
output.close()

"""m2 = GaussianFourier2d(rect, np.zeros((11,11)), 1.0, 1.0)
invProb2 = inverseProblem(fwd, m2, gamma, obspos, obs)
u0_fourier = mor.mapOnRectangle(rect, "fourier", np.zeros((11,11)))
uOpt_fourier = invProb2.find_uMAP(u0_fourier, nit=100, nfev=100)"""

#uList, uListUnique, PhiList = invProb.randomwalk_pCN(u0, 1000)
#u_new, u_new_mean, us, vals, vals_mean = invProb.EnKF(obs, 50, KL=False, N = 10, beta = 0.05)
#plt.figure(); plt.semilogy(PhiList)
#invProb.plotSolAndLogPermeability(uList[-1])
"""plt.figure();
plt.semilogy(vals)
plt.semilogy(vals_mean, 'r', linewidth=4)
invProb.plotSolAndLogPermeability(u_new_mean)"""


"""Is = []
hvals = np.linspace(-1,1,200)

for k in range(16):
	d = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(16, k)))
	Is.append([invProb.I(u_new_mean + d*h) for h in hvals])

plt.figure()
for k in range(16):
	plt.subplot(4,4,k+1)
	plt.plot(hvals, Is[k])"""
