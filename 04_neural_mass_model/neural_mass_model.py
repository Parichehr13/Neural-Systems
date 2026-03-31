 !/usr/bin/env python
  coding: utf-8

    Neural Mass Model
  
  Write a program to simulate the dynamics of three interacting populations (population of pyramidal neurons, excitatory interneurons and inhibitory interneurons) according to the model by Jansen and De Rit (1995). They used the following basic parameters of the model. 
  
  * Wep = 135
  * Wpe = 0.8*Wep
  * Wip = 0.25*Wep
  * Wpi = 0.25*Wep
  * ae = 3.25 mV
  * ai = 22 mV
  * 1/$\tau_e$ = 100s-1
  * 1/$\tau_i$ = 50 s-1
  
  Use the following expression for the sigmoid: $S(v) = rmax/( 1 + exp(-k*(v â€“ v0))$, with:
  * v0 = 6 mV
  * k = 0.56 mV-1
  * rmax = 5s-1
  
  Note that the model uses mV to indicate the variable leaving the synapse, since in the original work the output was defined as the post-synaptic potential.
  Other recommended parameters: mean value of 160 and standard deviation as great as 200 for the white noise; integration step of 1/10000.
  With the above parameters the model generates an alpha rhythm. Different rhythms can be obtained by varying the excitation time constant or the Wep parameter (and consequently the other synapses), e.g., Wep = 68, 108, 128, 270, 675, 1350.
  
  * Plot the EEG signal vs. time once the initial transient has expired.
  * Compute the power spectral density of the EEG signal and plot it vs. frequency (to improve the figure, do not plot low frequencies below 2 or 3 Hz).

  In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


  In[4]:


Wep = 135 
  135: alpha; 68 alpha-beta;  %108; % 128; %135 alpha;  %270 theta;   %675  % 1350
Wpe=0.8*Wep
Wip=0.25*Wep
Wpi=0.25*Wep
Ae=3.25
Ai=22  22;  %17.6;
ae=100  50 75 150 200
ai=50
kr=0.56
v0=6
rmax=5
dt=0.0001
tmax=200
t=np.arange(0, tmax+dt, dt)
N=len(t)
yp=np.zeros((N,))
zp=np.zeros((N,))
ye=np.zeros((N,))
ze=np.zeros((N,))
yi=np.zeros((N,))
zi=np.zeros((N,))

for k in np.arange(N-1):
    n=200*np.random.randn(1,1)+160

    vp=Wpe*ye[k]-Wpi*yi[k]
    ve=Wep*yp[k]
    vi=Wip*yp[k]
    rp=rmax/(1+np.exp(-kr*(vp-v0)))
    re=rmax/(1+np.exp(-kr*(ve-v0)))
    ri=rmax/(1+np.exp(-kr*(vi-v0)))
    dyp = zp[k]
    dzp = Ae*ae*rp-2*ae*zp[k]-ae*ae*yp[k]
    dye = ze[k]
    dze = Ae*ae*(re+n/Wpe)-2*ae*ze[k]-ae*ae*ye[k]
    dyi = zi[k]
    dzi = Ai*ai*ri-2*ai*zi[k]-ai*ai*yi[k]
    yp[k+1]=yp[k]+dyp*dt
    zp[k+1]=zp[k]+dzp*dt
    ye[k+1]=ye[k]+dye*dt
    ze[k+1]=ze[k]+dze*dt
    yi[k+1]=yi[k]+dyi*dt
    zi[k+1]=zi[k]+dzi*dt


  In[5]:


eeg = Wpe*ye-Wpi*yi
[f, Peeg] = welch(eeg, fs=1/dt, nperseg=20000, noverlap=10000) 


  In[7]:


start = int(1.01/dt)
stop= int(3/dt)

plt.figure(figsize=(11,8))
plt.plot(t[start:stop],eeg[start:stop],'k')
plt.ylabel('EEG amplitude')
plt.xlabel('time (s)')
plt.show()

plt.figure(figsize=(11,8))
plt.plot(f[5:80],Peeg[5:80],'k')
plt.ylabel('PSD')
plt.xlabel('frequency (Hz)')
plt.show()


  In[ ]:






