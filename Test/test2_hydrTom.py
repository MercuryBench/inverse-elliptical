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
rect = Rectangle((0,0), (2,1), resol=resol)
rectCoarse = Rectangle((0,0), (2,1), resol=resol-2)
gamma = 0.001


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
		return (-1*np.logical_and(y <= 0.2+0.25*x, True) + (-2)*np.logical_and(y>0.2+0.25*x, y<=2-x) + (1)*np.logical_and(y <= 0.2+0.25*x, y<= 1-2/3*x)+1)*3
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
f = []
f.append(mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-0.75)**2 + (y-0.2)**2 < 0.05**2)*(-20.0)))
f.append(mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-1.5)**2 + (y-0.3)**2 < 0.05**2)*(-20.0)))
f.append(mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-0.5)**2 + (y-0.7)**2 < 0.05**2)*(-20.0)))
f.append(mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-1.2)**2 + (y-0.1)**2 < 0.05**2)*(-20.0)))
f.append(mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-0.2)**2 + (y-0.4)**2 < 0.05**2)*(-20.0)))
f.append(mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-1.0)**2 + (y-0.8)**2 < 0.05**2)*(-20.0)))
f.append(mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-1.7)**2 + (y-0.2)**2 < 0.05**2)*(-20.0)))
f.append(mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-1.7)**2 + (y-0.8)**2 < 0.05**2)*(-20.0)))

fwdList = []
invProbList = []
	
m1 = GeneralizedGaussianWavelet2d(rect, 0.1, 1.5, resol)
for ff in f:
	fwdList.append(linEllipt2dRectangle(rect, ff, u_D, boundary_D_boolean))
	invProbList.append(inverseProblem(fwdList[-1], m1, gamma))
	
invProb = inverseProblem_hydrTom(rect, invProbList, m1)

#invProb2 = inverseProblem_hydrTom(rect, [invProbList[0],invProbList[2]], m1)



N_obs = 50

obsposx = np.random.uniform(0, 2, (N_obs,))
obsposy = np.random.uniform(0, 1, (N_obs,))
obspos = [obsposx, obsposy]
uTruth_temp = mor.mapOnRectangle(rectCoarse, "handle", lambda x, y: myUTruth(x, y))
uTruth = mor.mapOnRectangle(rect, "wavelet", uTruth_temp.waveletcoeffs)
for ip in invProbList:
	ip.obspos = obspos
	ip.obs = ip.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.plotSolAndLogPermeability(uTruth, obs=True, same=True)
#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
#uTruth = m1.sample()



u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean[0:rectCoarse.resol+1])
uOpt = u0
for kk in range(50):
	uOpt = invProb.find_uMAP(uOpt, nit=50, method = 'BFGS', adjoint=True, version=0);
	print(invProb.IList(uOpt))


invProb.plotSolAndLogPermeability(uOpt, same=True)#, save="testhydrTom.eps")

#data = {"obs": [obs , "obspos": obspos, "resol": resol, "gamma": gamma, "u0_waveletcoeffs": u0.waveletcoeffs, "uOpt_waveletcoeffs": uOptList[-1].waveletcoeffs}

"""u02 = mor.mapOnRectangle(rect, "wavelet", m1._mean[0:rectCoarse.resol])
uOpt2 = u02
for kk in range(50):
	uOpt2 = invProb2.find_uMAP(uOpt2, nit=150, method = 'BFGS', adjoint=True, version=0);
	print(invProb2.IList(uOpt2))"""
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



