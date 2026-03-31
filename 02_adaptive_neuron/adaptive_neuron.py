#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # Adaptive neuron - Integrate and fire with adaptation (with adaptation)
# 
# Modify the previous *integrate and fire model* (Stage 2) by including an adaptation conductance $g_a$. 
# 
# * Evaluate the response to a constant input current in the presence and absence of the adaptation. Use as an adaptation time constant in the range: $\tau_{a}$ = 300 - 1000 ms and an adaptation potential Ea = -90 mV. The adaptation conductance can be varied (example of a possible value: $r\cdot g_{a,max}$ = 1 - 5) to evaluate its effect. Choose a small value for $\Delta p_{a}=0.03-0.2$.
# 
# * **Optional**: Repeat the test with different constant current values, and obtain the â€œcurrent-discharge rateâ€ plot.
# 

# In[2]:


E0 = -65 #mV
Ea = -90 #mV
taum = 30 #ms (30-50ms)
taut = 10 #ms
taua = 1000 #ms (300-1000ms)
r = 10 #Mohm
C = taum/r

dPa = 0.1 #(0.03-0.2)
I = 4 #nA
Vtl = -55 #mV
Vth = 50 #mV
gamax = 2/r #gamax= (1-5)/r
g = 1/r 

dt = 0.01
tend = 800
t = np.arange(0, tend+dt, dt) #ms
L = len(t)
V = np.zeros((L,))
Vt = np.zeros((L,))
Pa = np.zeros((L,))

V[0] = -65 #mV
Pa[0] = 0
Vt[0] = Vtl


# In[3]:


idx_spike = []
for k in np.arange(L-1):
    ga = gamax * Pa[k]
    geq = g+ga
    E0tot = (g*E0+ga*Ea) /geq
    rtot = 1/geq
    Vinf = E0tot+rtot*I
    tau = C*rtot # updating membrane time constant
    V[k+1] = (V[k] - Vinf)*np.exp(-dt/tau) + Vinf
    Vt[k+1] = (Vt[k] - Vtl)*np.exp(-dt/taut) + Vtl
    Pa[k+1] = Pa[k]*np.exp(-dt/taua)  #Pa update between spikes
    if V[k + 1] > Vt[k + 1]:
        V[k + 1] = E0
        Vt[k + 1] = Vth
        Pa[k + 1] = Pa[k + 1] + dPa*(1 - Pa[k + 1]) #Pa update when spike arrived
        idx_spike.append(k+1)

spikes = np.zeros((L,))
spikes[idx_spike] = 1


# In[4]:


plt.figure(figsize=(11,8))
plt.subplot(2,2,1)
plt.plot(t,V, 'k', t,Vt,'r')
plt.legend(['V', 'Vt'], loc='upper right')
plt.xlabel('time (ms)')
plt.ylabel('(mV)')
plt.title('Potential')

plt.subplot(2,2,2)
plt.plot(t,Pa, 'k')
plt.ylim([0, 1])
plt.xlabel('time (ms)')
plt.title('Pa')

plt.subplot(2,2,3)
plt.plot(t,spikes, 'k')
plt.axis([0, t[-1], 0, 1.1])
plt.xlabel('time (ms)')
plt.title('Spikes')

plt.tight_layout()
plt.show()


# In[5]:


II = np.arange(0, 11, 0.5)
f = np.zeros(II.shape[0])
for trial in np.arange(len(II)):
    I = II[trial]
    
    V = np.zeros((L,))
    Vt = np.zeros((L,))
    Pa = np.zeros((L,))
    V[0] = -65 #mV
    Pa[0] = 0
    Vt[0] = Vtl
    idx_spike = []

    for k in np.arange(L-1):
        ga = gamax * Pa[k]
        geq = g+ga
        E0tot = (g*E0+ga*Ea) /geq
        rtot = 1/geq
        Vinf = E0tot+rtot*I
        tau = C*rtot
        V[k+1] = (V[k] - Vinf)*np.exp(-dt/tau) + Vinf
        Vt[k+1] = (Vt[k] - Vtl)*np.exp(-dt/taut) + Vtl
        Pa[k+1] = Pa[k]*np.exp(-dt/taua)
        if V[k + 1] > Vt[k + 1]:
            V[k + 1] = E0
            Vt[k + 1] = Vth
            Pa[k + 1] = Pa[k + 1] + dPa*(1 - Pa[k + 1])
            idx_spike.append(k+1)
    if len(idx_spike) > 1: # at least 2 spikes
        T = t[idx_spike[-1]]-t[idx_spike[-2]]  
        # frequency between the last two spikes; actually, the frequency changes over time.
        f[trial] = 1/T*1000
    else:
        f[trial] = 0
    #spikes = np.zeros((L,))
    #spikes[idx_spike] = 1
    
    #plt.figure(figsize=(11,8))
    #plt.subplot(1,2,1)
    #plt.plot(t,V,'k',t,Vt,'r')
    #plt.legend(['V', 'Vt'], loc='upper right')
    #plt.title('Potential')
    #plt.xlabel('time (ms)')
    #plt.ylabel('(mV)')
    #plt.axis([0, tend, E0-1, Vth+1])

    #plt.subplot(1,2,2)
    #plt.plot(t,Pa, 'k')
    #plt.title('Pa')

    #plt.tight_layout()
    #plt.show()


# In[6]:


plt.figure(figsize=(11,8))
plt.plot(II,f,'--k*')
plt.xlabel('Input current (nA)')
plt.ylabel('Frequency (Hz)')
plt.show()

# If you want to save the obtained f-i relationship and compare it with other
# models (Stages 1-2, save II and f variables also for Stages 1-2)

# from scipy.io import savemat
# savemat('if_discharge_rate_refr_period_adaptation.mat', {'II': II, 'f':f})

#or use np.savetxt(...)


# In[8]:


# If you want to compare f-i relationship with other models (Stages 1-3)

#from scipy.io import loadmat
#data = loadmat('if_discharge_rate.mat')
#II = np.squeeze(data['II'])
#f0 = np.squeeze(data['f'])
#data = loadmat('if_discharge_rate_refr_period.mat')
#f1 = np.squeeze(data['f'])
#data = loadmat('if_discharge_rate_refr_period_adaptation.mat')
#f2 = np.squeeze(data['f'])

#plt.figure(figsize=(11,8))
#plt.plot(II, f0,'--k*')
#plt.plot(II, f1,'--r*')
#plt.plot(II, f2,'--b*')

#plt.xlabel('Input current (nA)')
#plt.ylabel('Frequency (Hz)')
#plt.legend(['IF', 'IF+refr.period', 'IF+refr.period+adaptation'])
#plt.show()


# In[ ]:





