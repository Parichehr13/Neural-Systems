#!/usr/bin/env python
# coding: utf-8

# # Exercise on supervised learning: Training a network with one hidden layer through backpropagation
# 
# The response of peripheral chemoreceptors depends strongly non-linearly on the arterial pressure of oxygen and CO2. To derive the characteristic that binds the discharge frequency to these pressures, it is necessary to carry out highly invasive "open loop" experiments, in which one pressure is artificially kept constant and the other is modified in steps. In this exercise we propose to derive these characteristics directly from closed-loop data.
# 
# Consider the ‘chemo.mat’ file which contains values of PaO2 (‘Pao2’), PaCO2 (‘PaCo2’) and chemocector frequency (‘fac’). Train a two-layer neural network (1 output neuron + 1 hidden layer) to recognize the relationship between the firing rate and the pressures of O2 and CO2. As a first test, use only two neurons for the hidden layer, and sigmoidal features. The saturation value of the sigmoidal characteristics must be of the order of magnitude of the maximum frequency value in the experimental data.
# * Prove the convergence with different values of the learning constant (by plotting the error over the epochs).
# * Compare the values obtained from the network with the experimental ones (by plotting the network output and ground truth ‘fac’ values).
# * Check the network's ability to generalize. For this purpose, use the following theoretical formula, obtained from open-loop experiments:
# 
# fnor = 3 log(40/25)
# 
# fmax=12.3 #spikes/s
# 
# fmin=0.8352 #spikes/s
# 
# Pao2c=10
# 
# kc=75
# 
# k1=3
# 
# $fac=(k1\cdot log(Paco2/40)+fnor)/((Pao2-Pao2c)/kc)$
# 
# Choose a PaO2 value and compare the theoretical graph with that provided by the neural network at different PaCO2 values (that is, keeping PaO2 constant and varying PaCO2 in an assigned range). Repeat the test by setting PaCO2 and graph the output as the PaO2 varies.
# 
# * (optional) Implement the algorithm by imagining a number N of neurons in the hidden layer. Try training and generalization as the number of neurons in the hidden layer increases.
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os


# In[2]:


def sigmoid(u):
    s = 20/(1+np.exp(-u/10))
    return s

def sigder(u):
    s = np.exp(-u/10)/((1+np.exp(-u/10))**2)*20/10
    return s


# In[8]:


data = loadmat('chemo.mat')
Paco2 = data['Paco2']
Pao2 = data['Pao2']
fac = data['fac']

N = len(fac)

# hidden layer with 2 neurons 
W1o=np.random.rand(1)-0.5 #uniform distribution btw -0.5:0.5
W1c=np.random.rand(1)-0.5
teta1=np.random.rand(1)-0.5
W2o=np.random.rand(1)-0.5
W2c=np.random.rand(1)-0.5
teta2=np.random.rand(1)-0.5

# output layer with 1 neuron
Wu1=np.random.rand(1)-0.5
Wu2=np.random.rand(1)-0.5
tetau=np.random.rand(1)-0.5

gamma=0.001 #lr 
k=0
errs=[]
max_steps = int(os.getenv("EX10_MAX_STEPS", "80000"))
while k<max_steps:
    E = np.zeros((N,))
    for i in np.arange(N):
        # FORWARD
        # a) hidden activity
        u1=W1o*Pao2[i]+W1c*Paco2[i]-teta1
        u2=W2o*Pao2[i]+W2c*Paco2[i]-teta2 
        # b) activating hidden activity
        y1=sigmoid(u1)
        y2=sigmoid(u2)
        # c) output activity
        uusc=Wu1*y1+Wu2*y2-tetau
        # d) activating output activity
        usc=sigmoid(uusc)
        
        # ERRROR
        E[i]=fac[i]-usc
        
        # BACKPROP
        # output unit
        deltau=E[i]*sigder(uusc)
        DWu1=gamma*deltau*y1
        DWu2=gamma*deltau*y2
        Dtetau=gamma*deltau*(-1)
        # 1st hidden unit
        delta1=sigder(u1)*deltau*Wu1
        DW1o=gamma*delta1*Pao2[i]
        DW1c=gamma*delta1*Paco2[i]
        Dteta1=gamma*delta1*(-1)
        # 2nd hidden unit
        delta2=sigder(u2)*deltau*Wu2
        DW2o=gamma*delta2*Pao2[i]
        DW2c=gamma*delta2*Paco2[i]
        Dteta2=gamma*delta2*(-1)
        
        # UPDATE WEIGHTS
        Wu1=Wu1+DWu1
        Wu2=Wu2+DWu2
        tetau=tetau+Dtetau
        W1o=W1o+DW1o
        W1c=W1c+DW1c
        teta1=teta1+Dteta1
        W2o=W2o+DW2o
        W2c=W2c+DW2c
        teta2=teta2+Dteta2
    k+=1
    err=np.sum(np.array(E)**2)
    if k%100==0:
        errs.append(err)
        print("{0}k: err={1}".format(k/1000, err))


# In[10]:


# part 1
plt.figure(figsize=(11,8))
plt.plot((1+np.arange(len(errs)))*100, errs,'k',linewidth=2)
plt.xlabel('epochs',fontsize=14)
plt.ylabel('error',fontsize=14)
plt.show()


# In[12]:


# part 2
u1=W1o*Pao2+W1c*Paco2-teta1
u2=W2o*Pao2+W2c*Paco2-teta2
y1=sigmoid(u1)
y2=sigmoid(u2)
uusc=Wu1*y1+Wu2*y2-tetau
usc=sigmoid(uusc)

plt.figure(figsize=(11,8))
plt.plot(fac,'b*', markersize=10)
plt.plot(usc,'ro', markersize=10)
plt.xlabel('example')
plt.ylabel('fac')
plt.legend(['actual', 'predicted'])
plt.show()


# In[16]:


# part 3
K=3
B=25
fnor=K*np.log(40/B)
Pao2c=10
kc=75
k1=3 

Paco2=np.arange(40, 81, 5)
Pao2=50*np.ones((9,))

fac=(k1*np.log(Paco2/40)+fnor)/((Pao2-Pao2c)/kc)

u1=W1o*Pao2+W1c*Paco2-teta1
u2=W2o*Pao2+W2c*Paco2-teta2
y1=sigmoid(u1)
y2=sigmoid(u2)
uusc=Wu1*y1+Wu2*y2-tetau
usc=sigmoid(uusc)

plt.figure(figsize=(8,11))
plt.subplot(2,1,1)
plt.plot(Paco2,fac,'b*',Paco2,usc,'ro')
plt.xlabel('Paco2')
plt.ylabel('fac')
plt.legend(['empirical','predicted'])

Paco2=50*np.ones((16,))
Pao2=np.arange(20, 96, 5)

fac=(k1*np.log(Paco2/40)+fnor)/((Pao2-Pao2c)/kc)

u1=W1o*Pao2+W1c*Paco2-teta1
u2=W2o*Pao2+W2c*Paco2-teta2
y1=sigmoid(u1)
y2=sigmoid(u2)
uusc=Wu1*y1+Wu2*y2-tetau
usc=sigmoid(uusc)

plt.subplot(2,1,2)
plt.plot(Pao2,fac,'b*',Pao2,usc,'ro')
plt.xlabel('Pao2')
plt.ylabel('fac')
plt.tight_layout()
plt.show()


# In[ ]:



