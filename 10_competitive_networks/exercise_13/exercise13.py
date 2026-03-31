#!/usr/bin/env python
# coding: utf-8

# # Exercises on competitive networks: Simulation of a neural network with lateral inhibition
# 
# Create a neural network composed of N=180 neurons arranged in a one-dimensional chain. Each neuron receives one input from the outside (e.g., a light beam in the compound eye). Furthermore, each neuron receives a self-excitation, and an inhibition from each of the other neurons. The strength of the inhibitory synapse decreases with the distance between neurons, through a Gaussian function (e.g., with a standard deviation of 24). 
# 
# * Simulate the dynamic of the network using Euler's method (e.g., with a step of 0.1 s). Assume 1st order dynamics for each neuron, with a time constant of 3 s. Set neurons' thresholds at 6, the self-excitatory synapses at lex0=5 and the lateral inhibitory synapses at lin0=2. Observe how the behavior of the system changes as the value of the synapses varies (i.e., by varying the value of the self-excitatory synapse and the exponential law of lateral inhibition). Suppose that neurons have a sigmoidal excitation function. For simplicity, assume maximum activity as great as to one; furthermore, use a slope term of 0.6. Use as input signal (for example varying between 5 and 15):
#     * a rectangular stimulus (contrast enhancement).
#     * a stimulus comprising two nearby stimuli modelled via Gaussian functions and immersed in a high background (improved resolution).
#     
# To better understand the network output in these two cases, you can also plot the output of the network without competition (i.e., the output related to the input only). 
# 
# * (optional) Modify the previous exercise by imagining a circular law for lateral synapses, in order to  avoid edge effects. Simulate the network as in the previous point. 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def sigmoid(x, k=0.6, x0=0):
    # s(x) = 1 / 1 + exp(-k(x-x0))
    tmp = -k*(x-x0)
    return 1/(1+np.exp(tmp))


# In[3]:


N=180              
sigin=24 # 6                                                
Lex0=5 # 2.4 #6                                                               
Lin0=2 # 1.4 #3

type_structure = 'circular' # or 'circular'

L = np.zeros((N, N))
k = np.arange(N)
for i in k:
    if type_structure == 'linear':
        d = np.abs(k-i)
    elif type_structure == 'circular':
        d = np.abs(k-i)
        idx = np.where(d > N/2)[0]
        d[idx] = N - d[idx]
    L[i,:] = -Lin0 * np.exp(-d**2/(2*sigin**2))
    L[i,i] = Lex0


# In[4]:


stim_type = 1
baseline_value = 5
if stim_type==0:
    # rectangular stimulus
    high_value = 10
    Ix = baseline_value*np.ones((N,))
    idx_start = round(N/2)
    idx_stop = idx_start + round(N/4)
    Ix[idx_start:idx_stop] = high_value
    
elif stim_type==1:
    # 2 narrow stimuli
    Ix = baseline_value*np.ones((N,))
    
    idx_stim0 = 100#85
    idx_stim1 = 120#115
    siginput = 7#10
    
    Ix = Ix + 10*np.exp(-(np.arange(N)-idx_stim0)**2/(2*siginput**2))            + 10*np.exp(-(np.arange(N)-idx_stim1)**2/(2*siginput**2))


# In[5]:


sigm_k=0.6
threshold=6
tau = 3
max_iter = 1000
dt = 0.1  
t = np.arange(max_iter) * dt 
x = np.zeros((N,max_iter))

#its = []
plt.figure(figsize=(11,8))
for k in np.arange(len(t) - 1):
    tmp = Ix + np.matmul(L, x[:,k])-threshold
    x[:,k+1] = x[:,k] + dt*( (1/tau) * (-x[:,k] + sigmoid(tmp, k=sigm_k)) ) 
    if k % 50 == 0:
        plt.plot(np.arange(N), x[:,k+1], 'k', linewidth=1)
        plt.axis([0, 200, 0, 1])
        #its.append('step '+ str(k))
plt.xlabel('neuron')
plt.ylabel('x')
#plt.legend(its)
plt.show()

feedforward = sigmoid(Ix-threshold, k=sigm_k)

plt.figure(figsize=(11,8))
plt.subplot(2,1,1)
plt.plot(np.arange(N),Ix,'g')
plt.xlabel('neuron')
plt.ylabel('input')
plt.subplot(2,1,2)
plt.plot(np.arange(N),x[:,k+1],'r',np.arange(N),feedforward,'b',linewidth=2)
plt.xlabel('neuron')
plt.ylabel('x')
plt.tight_layout()
plt.show()


# In[ ]:




