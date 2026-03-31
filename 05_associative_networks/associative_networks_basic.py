 !/usr/bin/env python
  coding: utf-8

  In[1]:


import numpy as np
import matplotlib.pyplot as plt


    Hetero-associative networks
  
  Build three binary vectors (with values Â± 1) of 10 elements. Each of them represents a possible input pattern. Normalize the vectors so that each of them has unitary norm and arrange them along three columns of an X matrix. Consider a hetero-associative network with 10 inputs and 3 output neurons. The output neurons have activation values between 0 and 1.
  
  Train the network with Hebb's rule, so that at the first input pattern (= first column of X) only the first output neuron is active, at the second input pattern only the second neuron is active, and in the presence of the third input pattern only the third neuron is active (therefore the activation of the first three neurons corresponds to the recognition of the relative pattern).
  
  Give one of the three patterns of your choice as input to the network, corrupted by the addition of Gaussian noise with a null mean value and assigned variance (after noise addiction the vector must be normalized again). Calculate the network output:
  * assuming linear neurons.
  * assuming that neurons have a sigmoidal function with output between 0 and 1 and central value = 0.5. Consider the possibility of choosing the steepness of slope of the sigmoid. Study the performance of the network as the slope of the sigmoid varies.
  

  In[2]:


  inputs
X_no_noise = 2*np.round(np.random.rand(10,3)) - 1
X_no_noise = X_no_noise/np.sqrt(10)

  inputs with noise
sigma = 0.1
X_noise = X_no_noise + sigma*np.random.randn(10,3)
X_noise[:,0]= X_noise[:,0]/np.linalg.norm(X_noise[:,0])
X_noise[:,1]= X_noise[:,1]/np.linalg.norm(X_noise[:,1])
X_noise[:,2]= X_noise[:,2]/np.linalg.norm(X_noise[:,2])

Y = np.eye(3,3)
  network training (on inputs without noise)
W = np.matmul(Y, X_no_noise.T)

  network output (without noise)
Y1_no_noise = np.matmul(W, X_no_noise[:,0].reshape((X_no_noise.shape[0], 1)))
Y2_no_noise = np.matmul(W, X_no_noise[:,1].reshape((X_no_noise.shape[0], 1)))
Y3_no_noise = np.matmul(W, X_no_noise[:,2].reshape((X_no_noise.shape[0], 1)))

  network output (with noise)
Y1_noise = np.matmul(W, X_noise[:,0].reshape((X_noise.shape[0], 1)))
Y2_noise = np.matmul(W, X_noise[:,1].reshape((X_noise.shape[0], 1)))
Y3_noise = np.matmul(W, X_noise[:,2].reshape((X_noise.shape[0], 1)))

  sigmoid-activated network output - k = 10
k = 10
Y1_sig_no_noise = 1./(1+np.exp(-k*(Y1_no_noise - 0.5)))
Y2_sig_no_noise = 1./(1+np.exp(-k*(Y2_no_noise - 0.5)))
Y3_sig_no_noise = 1./(1+np.exp(-k*(Y3_no_noise - 0.5)))

Y1_sig_noise = 1./(1+np.exp(-k*(Y1_noise - 0.5)))
Y2_sig_noise = 1./(1+np.exp(-k*(Y2_noise - 0.5)))
Y3_sig_noise = 1./(1+np.exp(-k*(Y3_noise - 0.5)))

  sigmoid-activated network output - k = 20
k = 20
Y1_sig_no_noise_ = 1./(1+np.exp(-k*(Y1_no_noise - 0.5)))
Y2_sig_no_noise_ = 1./(1+np.exp(-k*(Y2_no_noise - 0.5)))
Y3_sig_no_noise_ = 1./(1+np.exp(-k*(Y3_no_noise - 0.5)))

Y1_sig_noise_ = 1./(1+np.exp(-k*(Y1_noise - 0.5)))
Y2_sig_noise_ = 1./(1+np.exp(-k*(Y2_noise - 0.5)))
Y3_sig_noise_ = 1./(1+np.exp(-k*(Y3_noise - 0.5)))


  In[3]:


def compare_neuron_values(Y_no_noise, Y_noise, Y_sig_no_noise, Y_sig_noise, Y_sig_no_noise_, Y_sig_noise_):
      auxillary function to plot network output as bars
    plt.figure(figsize=(11,7))
    for i in np.arange(3):
        plt.subplot(1,3,i+1)
        plt.bar(np.arange(6),
                [Y_no_noise[i ,0], 
                 Y_noise[i ,0], 
                 Y_sig_no_noise[i ,0], 
                 Y_sig_noise[i ,0],
                 Y_sig_no_noise_[i ,0], 
                 Y_sig_noise_[i ,0]],
               width=0.25, facecolor='k')
        plt.ylim([-1, 1])
        plt.axhline(y=0, c='r')
        plt.xticks(np.arange(6), ['no noise',
                   'noise',
                   'sigmoided (k=10) no noise',
                   'sigmoided (k=10) noise',
                   'sigmoided (k=20) no noise',
                   'sigmoided (k=20) noise',], rotation=45, 
                  horizontalalignment='right')
        plt.title('Neuron: {0}'.format(i))
    plt.tight_layout()
    plt.show()
    
print(' '*10+' First input')
print('linear-activated without noise:', Y1_no_noise)
print('linear-activated with noise:', Y1_noise)
print('sigmoid-activated (k=10) without noise:', Y1_sig_no_noise)
print('sigmoid-activated (k=10) with noise:', Y1_sig_noise)
print('sigmoid-activated (k=20) without noise:', Y1_sig_no_noise_)
print('sigmoid-activated (k=20) with noise:', Y1_sig_noise_)
compare_neuron_values(Y1_no_noise, Y1_noise, 
                      Y1_sig_no_noise, Y1_sig_noise,
                      Y1_sig_no_noise_, Y1_sig_noise_) 
print(' '*10+' Second input')
print('linear-activated without noise:', Y2_no_noise)
print('linear-activated with noise:', Y2_noise)
print('sigmoid-activated (k=10) without noise:', Y2_sig_no_noise)
print('sigmoid-activated (k=10) with noise:', Y2_sig_noise)
print('sigmoid-activated (k=20) without noise:', Y2_sig_no_noise_)
print('sigmoid-activated (k=20) with noise:', Y2_sig_noise_)
compare_neuron_values(Y2_no_noise, Y2_noise, 
                      Y2_sig_no_noise, Y2_sig_noise,
                      Y2_sig_no_noise_, Y2_sig_noise_)  
print(' '*10+' Third input')
print('linear-activated without noise:', Y3_no_noise)
print('linear-activated with noise:', Y3_noise)
print('sigmoid-activated (k=10) without noise:', Y3_sig_no_noise)
print('sigmoid-activated (k=10) with noise:', Y3_sig_noise)
print('sigmoid-activated (k=20) without noise:', Y3_sig_no_noise_)
print('sigmoid-activated (k=20) with noise:', Y3_sig_noise_)
compare_neuron_values(Y3_no_noise, Y3_noise, 
                      Y3_sig_no_noise, Y3_sig_noise,
                      Y3_sig_no_noise_, Y3_sig_noise_) 


  In[ ]:






