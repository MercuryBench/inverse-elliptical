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

xticks_port = np.array([27.5, 48, 68.4,93.1,114,134])
yticks_port = np.array([8.33, 16.5, 24.8, 33, 41.3, 49.6, 57.8, 66.1])

ts = [(x, y) for x in xticks_port for y in yticks_port]
num = len(ts)
l = np.zeros([num, 2])
for n, t in enumerate(ts):
	l[n, 0] = t[0]
	l[n, 1] = t[1]
#XX, YY = np.meshgrid(xticks_port, yticks_port)

"""plt.figure(); plt.ion()
plt.scatter(l[:, 0], l[:, 1])
plt.xlim([0, 160])
plt.ylim([0, 78])
plt.show()"""

resol = 8
rect = Rectangle((0.0,0.0), (160.0,78.0), resol=resol)
rectCoarse = Rectangle((0.0,0.0), (160.0,78.0), resol=resol-3)
gamma = 1.0

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
		return -1*np.logical_and(y <= 10+3/16*x, True) +  (-4)*np.logical_and(y <= 60-0.25*x, y > 10 + 3/16*x) + (2)*np.logical_and(y>60-0.25*x, np.logical_and(y<=0.8*x, y > 10+3/16*x))
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

obsposx = l[:,0]
obsposy = l[:,1]

import functools


obsindlist = [8, 29, 13, 33, 42, 28, 18, 15, 1, 22, 6, 24,  27, 39, 20]
partiallist = [functools.partial(lambda x, y, t: ((x-t[0])**2 + (y-t[1])**2 < 1.0**2)*(-20.0), t=ts[ind]) for ind in obsindlist]
flist = [mor.mapOnRectangle(rect, "handle", lambda x, y: f(x, y)) for f in partiallist]

f1 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-48.0)**2 + (y-8.33)**2 < 1.0**2)*(-20.0))
f2 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-93.1)**2 + (y-49.6)**2 < 1.0**2)*(-20.0))
f3 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-48)**2 + (y-49.6)**2 < 1.0**2)*(-20.0))
f4 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-114)**2 + (y-16.5)**2 < 1.0**2)*(-20.0))
f5 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-134)**2 + (y-24.8)**2 < 1.0**2)*(-20.0))
f6 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-93.1)**2 + (y-41.3)**2 < 1.0**2)*(-20.0))
f7 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-68.4)**2 + (y-24.8)**2 < 1.0**2)*(-20.0))
f8 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-48)**2 + (y-66.1)**2 < 1.0**2)*(-20.0))

f9 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-27.5)**2 + (y-16.5)**2 < 1.0**2)*(-20.0))
f10 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-68.4)**2 + (y-57.8)**2 < 1.0**2)*(-20.0))

f11 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-27.5)**2 + (y-57.8)**2 < 1.0**2)*(-20.0))
f12 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-93.1)**2 + (y-8.33)**2 < 1.0**2)*(-20.0))
f13 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-93.1)**2 + (y-33)**2 < 1.0**2)*(-20.0))
f14 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-114)**2 + (y-66.1)**2 < 1.0**2)*(-20.0))
f15 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-68.4)**2 + (y-41.3)**2 < 1.0**2)*(-20.0))
f16 = mor.mapOnRectangle(rect, "handle", lambda x, y: ((x-134)**2 + (y-57.8)**2 < 1.0**2)*(-20.0))

f = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15] #8, 30, 14

fwdList = []
invProbList = []
	
m1 = GeneralizedGaussianWavelet2d(rect, 0.1, 1.5, resol)
for ff in f:
	fwdList.append(linEllipt2dRectangle(rect, ff, u_D, boundary_D_boolean))
	invProbList.append(inverseProblem(fwdList[-1], m1, gamma))
	
invProb = inverseProblem_hydrTom(rect, invProbList, m1)


#fwd = linEllipt2dRectangle(rect, f, u_D, boundary_D_boolean)

#invProb = inverseProblem(fwd, m1, gamma)
	


#invProb2 = inverseProblem_hydrTom(rect, [invProbList[0],invProbList[2]], m1)



N_obs = len(obsposx) -1

#obsposx = np.random.uniform(0, 2*scale, (N_obs,))
#obsposy = np.random.uniform(0, 1, (N_obs,))
obspos = [obsposx, obsposy]

uTruth_temp = mor.mapOnRectangle(rectCoarse, "handle", lambda x, y: myUTruth(x, y))
uTruth = mor.mapOnRectangle(rect, "wavelet", uTruth_temp.waveletcoeffs)

forbidden = [8, 29, 13, 33, 42, 28, 18, 15, 1, 22, 6, 24,  27, 39, 20]

for m, ip in enumerate(invProbList):
	bx = [x for i,x in enumerate(obsposx) if i!=forbidden[m]]
	by = [x for i,x in enumerate(obsposy) if i!=forbidden[m]]
	ip.obspos = [bx, by]
	ip.obs = ip.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
invProb.plotSolAndLogPermeability(uTruth, obs=True, same=True)

#invProb.obspos = obspos
#invProb.obs = invProb.Gfnc(uTruth) + np.random.normal(0, gamma, (N_obs,))
#invProb.plotSolAndLogPermeability(uTruth, obs=invProb.obs)

#uTruth = mor.mapOnRectangle(rect, "wavelet", packWavelet(unitvec(4, 2)+0.5*unitvec(4,3)))
#uTruth = m1.sample()


u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean[0:rectCoarse.resol+1])
uOpt = u0
for kk in range(3):
	uOpt = invProb.find_uMAP(uOpt, nit=360, method = 'BFGS', adjoint=True, version=0);
	invProb.plotSolAndLogPermeability(uOpt, obs=True, same=True)
	print(invProb.IList(uOpt))

#D_dual_exact = invProb.DI_adjoint_vec_wavelet(uOpt, version=2)
#D_dual = invProb.DI_adjoint_vec_wavelet(uOpt, version=0)

plt.figure()
plt.plot(D_dual_exact, 'g')
plt.plot(D_dual, 'r')
#plt.plot(D_dual*160*78, 'r')
"""u0 = mor.mapOnRectangle(rect, "wavelet", m1._mean[0:rectCoarse.resol+1])
uOpt = u0
for kk in range(150):
	uOpt = invProb.find_uMAP(uOpt, nit=250, method = 'BFGS', adjoint=True, version=0);
	


invProb.plotSolAndLogPermeability(uOpt, obs=invProb.obs)#, save="testhydrTom.eps")
"""
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
data = {"obs": [ip.obs for ip in invProb.invProbList], "obspos": obspos, "resol": resol, "gamma": gamma, "uTruth_waveletcoeffs": uTruth.waveletcoeffs, "uOpt_waveletcoeffs": uOpt.waveletcoeffs}
savefile = "data_hydrTom.pkl"
output = open(savefile, 'wb')
pickle.dump(data, output)
print("Saved as " + savefile)"""

"""data = unpickleData("data_hydrTom.pkl")
obspos = data["obspos"]
obsList = data["obs"]
for n in range(len(obsList)):
	invProb.invProbList[n].obs = obsList[n]


gamma = data["gamma"]
resol = data["resol"]
uTruth = mor.mapOnRectangle(rect, "wavelet", data["uTruth_waveletcoeffs"])
uOpt = mor.mapOnRectangle(rect, "wavelet", data["uOpt_waveletcoeffs"])
invProb.plotSolAndLogPermeability(uOpt, same=True, obs=True)"""


