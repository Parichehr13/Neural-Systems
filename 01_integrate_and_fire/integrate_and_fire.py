#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # Stage 1 - Integrate and fire neuron (with constant threshold value)
# Consider the model of a single *integrate and fire* neuron, consisting of a capacitance C, a conductance g the resting potential E0 = Vreset, driven by a current i(t) (order of magnitude of the current 0-4 nA). Plot the membrane potential V, and the individual output spikes, using a variable current as input (for example, a of rectified sinusoids or a triangular shape). Both the existence of refractory time and adaptation are neglected. Other parameters: E0 = Vreset = - 65 mV; Vth = -50 mV; $\tau_{m}$ = 30 - 50 ms; r = 10 $M \Omega$.

# In[2]:


# Setting up parameters
E0=-65  # resting potential (mV)
tau=30  #time constant (ms)
r=10  # membrane resistance (MOhm)
g=1/r  # membrane conductance
C=tau/r
Vt=-55  # threshold potential (mV)

dt=0.01
tend = 300 # ms
t=np.arange(0, tend+dt, dt)
L=len(t)

V=np.zeros((L, ))
V[0]=-65  # setting up initial value (mV)

Imax = 4.0  # maximum current amplitude (nA)
I = np.abs(Imax*np.sin(np.pi*t/tend))  #np.abs(Imax*np.sin(np.pi*t/tend))


# In[3]:


idx_spike=[]
for k in np.arange(L-1):
    Vinf = E0 + r*I[k]
    # V[k+1] = (V[k] - Vinf)*np.exp(-dt/tau) + Vinf # exploiting analytical solution assuming constant current at each step
    V[k+1] = V[k] + (Vinf - V[k])*dt/tau # Euler's method 
    if V[k+1]> Vt:
        # potential over threshold: reset value and count spikes
        V[k+1]=E0
        idx_spike.append(k+1)
spikes = np.zeros((L,))
spikes[idx_spike] = 1  # spikes train

plt.figure(figsize=(11,8))
plt.plot(t,I,'k')
plt.xlabel('time (ms)')
plt.ylabel('I (nA)')
plt.title('Input current')

plt.figure(figsize=(11,8))
plt.plot(t,V,'k')
plt.plot(t, Vt*np.ones(L,), 'r--')
plt.legend(['V', 'Vt'])
plt.ylabel('(mV)')
plt.xlabel('time (ms)')
plt.title('Potential')

plt.figure(figsize=(11,8))
plt.plot(t,spikes, 'k')
plt.xlabel('time (ms)')
plt.title('Spikes')
plt.show()


# In[6]:


II = np.arange(0, 11, 0.5) # constant current values (nA) to study
n_trials = len(II)
f = np.zeros((n_trials, ))

for trial in np.arange(n_trials):
    V=np.zeros((L, ))
    V[0] = -65  #mV
    idx_spike = []
    I = II[trial]
    
    for k in np.arange(L-1):
        Vinf = E0 + r*I
        V[k+1] = (V[k] - Vinf) * np.exp(-dt/tau) + Vinf
        if V[k+1] > Vt:
            V[k+1] = E0
            idx_spike.append(k+1)

    if len(idx_spike) > 1:
        T = t[idx_spike[-1]]-t[idx_spike[-2]]
        f[trial]=1/T*1000
    else:
        f[trial] = 0

    spikes = np.zeros((L,))
    spikes[idx_spike] = 1
    
    plt.figure(figsize=(11,8))
    plt.subplot(1,2,1)
    plt.plot(t,V,'k',t,Vt*np.ones(L,),'r')
    plt.legend(['V', 'Vt'], loc='upper right')
    plt.title('Potential')
    plt.xlabel('time (ms)')
    plt.ylabel('(mV)')
    plt.axis([0, tend, E0-1, Vt+1])

    plt.subplot(1,2,2)
    plt.plot(t,spikes, 'k')
    plt.title('Spikes')
    plt.xlabel('time (ms)')
    plt.axis([0, t[-1], 0, 1.1])
    
    plt.suptitle('Trial no. {0} (const. input current of {1} nA)'.format(trial, I))
    plt.show()
    
plt.figure(figsize=(11,8))
plt.plot(II,f,'--k*')
plt.xlabel('Input current (nA)')
plt.ylabel('Frequency (Hz)')
plt.title('Integrate and fire model with constant threshold potential')
plt.show()


# # Stage 2 - Integrate and fire neuron (with variable threshold value)
# Modify the previous model by including the relative refractory period by means of a variable threshold. 
# 1. Evaluate the response to a constant input current (e.g., i = 4 nA). 
# 2. Repeat the test with different constant current values (e.g., from 0 nA to 11 nA), and obtain the graph *discharge current-frequency*, point by point.
# 
# Other recommended parameters: VtL = -55 mV; VtH = 0 mV; $\tau_{t}$ = 10 ms and see parameters of Stage 1.

# In[7]:


E0=-65  #mV
tau=30  #ms (10-30ms)
taut=10  #ms   10 20 30
r=10  #Mohm

dt=0.01
tend = 300
t=np.arange(0, tend+dt, dt)
L=len(t)

Vtl=-55 #mV
Vth= 0  #mV   -55  0 30
g=1/r

II = np.arange(0, 11, 0.5) # constant current values (nA) to study


# In[8]:


V=np.zeros((L, ))
Vt=np.zeros((L, ))
V[0] = -65  #mV
Vt[0] = Vtl
idx_spike = []
I = II[5] # selecting one of the input currents available
    
for k in np.arange(L-1):
    Vinf = E0 + r*I
    V[k+1] = (V[k] - Vinf) * np.exp(-dt/tau) + Vinf
    Vt[k+1] = (Vt[k] - Vtl) * np.exp(-dt/taut) + Vtl
    if V[k+1] > Vt[k+1]:
        V[k+1] = E0
        Vt[k+1] = Vth
        idx_spike.append(k+1)

spikes = np.zeros((L,))
spikes[idx_spike] = 1
    
plt.figure(figsize=(11,8))
plt.subplot(1,2,1)
plt.plot(t,V,'k',t,Vt,'r')
plt.legend(['V', 'Vt'])
plt.title('Potential')
plt.xlabel('time (ms)')
plt.ylabel('(mV)')
plt.axis([0, tend, E0-1, Vth+1])

plt.subplot(1,2,2)
plt.plot(t,spikes, 'k')
plt.title('Spikes')
plt.xlabel('time (ms)')
plt.axis([0, t[-1], 0, 1.1])

plt.show()


# In[9]:


n_trials = len(II)
f = np.zeros((n_trials, ))

for trial in np.arange(n_trials):
    V=np.zeros((L, ))
    Vt=np.zeros((L, ))
    V[0] = -65  #mV
    Vt[0] = Vtl
    idx_spike = []
    I = II[trial]
    
    for k in np.arange(L-1):
        Vinf = E0 + r*I
        V[k+1] = (V[k] - Vinf) * np.exp(-dt/tau) + Vinf
        Vt[k+1] = (Vt[k] - Vtl) * np.exp(-dt/taut) + Vtl
        if V[k+1] > Vt[k+1]:
            V[k+1] = E0
            Vt[k+1] = Vth
            idx_spike.append(k+1)

    if len(idx_spike) > 1:
        T = t[idx_spike[-1]]-t[idx_spike[-2]]
        f[trial]=1/T*1000
    else:
        f[trial] = 0

    spikes = np.zeros((L,))
    spikes[idx_spike] = 1
    
    plt.figure(figsize=(11,8))
    plt.subplot(1,2,1)
    plt.plot(t,V,'k',t,Vt,'r')
    plt.legend(['V', 'Vt'], loc='upper right')
    plt.title('Potential')
    plt.xlabel('time (ms)')
    plt.ylabel('(mV)')
    plt.axis([0, tend, E0-1, Vth+1])

    plt.subplot(1,2,2)
    plt.plot(t,spikes, 'k')
    plt.title('Spikes')
    plt.xlabel('time (ms)')
    plt.axis([0, t[-1], 0, 1.1])
    
    plt.suptitle('Trial no. {0} (const. input current of {1} nA)'.format(trial, I))
    plt.show()
    
plt.figure(figsize=(11,8))
plt.plot(II,f,'--k*')
plt.xlabel('Input current (nA)')
plt.ylabel('Frequency (Hz)')
plt.title('Integrate and fire model with variable threshold potential')
plt.show()


# In[ ]:





# In[ ]:





