#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path


# # Exercises on the Hopfield model
# 
# Test the following changes to the Hopfield model.
# * Dilution. Assume, with reference to the exercise 8 (i.e., using the same patterns and training as with exercise 8), that a certain percentage of the synapses is damaged or missing. To this end, assign a value of 0 to a certain percentage (perc) of synapses. Simulate the behavior of the network for different dilution values and find out how high the dilution must be in order loose stored patterns in recovery. 
# * Low M/N ratio. Simulate the learning of 4 images of 6x6 size (so a total of 36 neurons) in a Hopfield network, for example by generating 4 letters of the alphabet. The position of the letters will strongly affect the level of correlation between the patterns. Examine the frequency with which spurious patterns emerge. 
# * Sparse patterns. Consider a Hopfield network with values 0 and 1 for neurons. Suppose that the information is "sparse", that is, only a percentage *a* of neurons (with *a* low, for example $a = 0.02$ or $a = 0.05$) is at the value +1 in the stored patterns. Three images with sparse coding are then generated (for example 18 pixels at +1 in a 30x30 image). Adopt the following Hebb rule:$\Delta W_{ij}=(y_i - a)(y_j - a)$. Modify the threshold of neurons (same threshold for all neurons) and look for the threshold value that guarantees good functioning. The rule for identifying neurons that can switch, and the formula for neuron switching, need to be modified accordingly.
# 
# 
# As with exercise 8, it is suggested to implement and use auxiliary functions to transform images into arrays and vice versa (from_mtx_to_array and from_array_to_mtx), and to binarize images (im2bw). The latter function is useful only for the first point of exercise 9. 
# 
# 

# In[2]:


# defining auxiliary functions
def from_mtx_to_array(I):
    # I: ndarray (N,N)
    nrows = I.shape[0]
    V = []
    for i in np.arange(nrows):
        V.extend(I[i,:])
    return np.array(V)

def from_array_to_mtx(V):
    # V: ndarray (N**2)
    nrows = int(np.sqrt(V.size))
    I = np.zeros((nrows, nrows))
    for i in np.arange(nrows):
        start = i*nrows
        stop = i*nrows+nrows
        I[i,:] = V[start:stop] #V(1+(j-1)*N:j*N);
    return I
def im2bw(I, th_value=128):
    #th_value depends on the specific data type (uint8: 0-255 values)
    I_thresholded = np.zeros_like(I)
    I_thresholded[I>=th_value] = 1
    return I_thresholded


# In[3]:


# Dilution
data_path = Path(__file__).resolve().with_name('imdemos.mat')
if not data_path.exists():
    data_path = Path(__file__).resolve().parent.parent / 'exercise08' / 'imdemos.mat'
data = loadmat(str(data_path))
#   box              128x128            16384  logical              
#   circles          256x256            65536  logical              
#   circuit          128x128            16384  uint8                
#   circuit4         256x256            65536  uint8                
#   coins            128x128            16384  uint8                
#   coins2           256x256            65536  uint8                
#   dots             128x128            16384  logical              
#   eight            256x256            65536  uint8                
#   glass            128x128            16384  uint8                
#   glass2           256x256            65536  uint8                
#   liftbody128      128x128            16384  uint8                
#   liftbody256      256x256            65536  uint8                
#   moon             128x128            16384  uint8                
#   pepper           128x128            16384  uint8                
#   pout             128x128            16384  uint8                
#   quarter          128x128            16384  uint8                
#   rice             128x128            16384  uint8                
#   rice2            128x128            16384  uint8                
#   rice3            256x256            65536  uint8                
#   saturn           128x128            16384  uint8                
#   saturn2          256x256            65536  uint8                
#   tire             128x128            16384  uint8                
#   trees            128x128            16384  uint8                
#   vertigo          128x128            16384  uint8                
#   vertigo2         256x256            65536  uint8  

XX1 = data['saturn']
XX2 = data['vertigo']
XX3 = data['coins']

XX1 = im2bw(XX1)
XX2 = im2bw(XX2)
XX3 = im2bw(XX3)

XX1=(XX1-0.5)*2
XX2=(XX2-0.5)*2
XX3=(XX3-0.5)*2

X1=XX1[::2,::2]
X2=XX2[::2,::2]
X3=XX3[::2,::2]

plt.figure(figsize=(11,5))
plt.subplot(1,3,1)
plt.imshow(X1, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(X2, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(X3, cmap='gray')
plt.show()


Y1 = from_mtx_to_array(X1)
Y2 = from_mtx_to_array(X2)
Y3 = from_mtx_to_array(X3)

Y = np.copy(Y1)
# initial state (switching a small percentage of neurons of one image)
perc=0.05
N=len(Y1)
idx_to_switch = np.where(np.random.rand(N,)<perc)
Y[idx_to_switch]=-Y[idx_to_switch]

# training
Y1 = Y1.reshape((Y1.shape[0], 1))
Y2 = Y2.reshape((Y2.shape[0], 1))
Y3 = Y3.reshape((Y3.shape[0], 1))
W = np.matmul(Y1, Y1.T)+np.matmul(Y2, Y2.T)+np.matmul(Y3, Y3.T)

# removing a percentage (assigned) of synapses
diluition = 0.997 #0.995, 0.998
# mask_damage is a matrix with the same shape of W with 0 or 1. 
# It will put randomly to 0 some weights in the weight matrix, simulating a damage in the network
mask_damage = (np.random.rand(N,N) > diluition).astype(int) 
W = W * mask_damage

# find the list of neurons that can switch (L neurons)
idx_neurons_to_switch = np.where(Y*np.matmul(W, Y)<0)[0]
L=len(idx_neurons_to_switch)
print('The number of neurons to switch (L) is: ', L)
# visualize initial perturbate state
plt.figure(figsize=(5,5))
plt.imshow(from_array_to_mtx(Y), cmap='gray')
plt.title('Initial perturbated image')
plt.show()

while L > 0:# until the number of neurons to switch (L) is = 0 
    # step 0: pick one random integer from 0 to L-1
    idx = np.random.randint(L)
    
    # step 1: switch that neuron (1 neuron for each step is switched)
    Y[idx_neurons_to_switch[idx]] = -1 * Y[idx_neurons_to_switch[idx]]
    
    # step 2: as the neuron value changed, the other neurons changed their value too,
    # so we recompute the list of neurons to switch (update of the list of neurons to switch)
    idx_neurons_to_switch = np.where(Y*np.matmul(W, Y)<0)[0]
    
    L=len(idx_neurons_to_switch) # update variable L 
    if L % 25 == 0: # visualize output only after 25 iterations
        print('The number of neurons to switch (L) is: ', L)
        plt.figure(figsize=(5,5))
        plt.imshow(from_array_to_mtx(Y), cmap='gray')
        plt.show()


# In[4]:


# Low M / N ratio
N1 = 6 
I1 = -np.ones((N1,N1)) 
I1[0,1:] = 1
I1[1:5,3] = 1 
I2 = -np.ones((N1,N1)) 
I2[:5,0] = 1
I2[:5,3] = 1
I2[2,:4] = 1
I3 = -np.ones((N1,N1))
I3[:5,1] = 1 
I3[4,1:5] = 1
I4 = -np.ones((N1,N1))
I4[0,2:]=1
I4[0:5,2]=1
I4[4,2:]=1

plt.figure(figsize=(5,5))
plt.subplot(2,2,1)
plt.imshow(I1, cmap='gray')
plt.subplot(2,2,2)
plt.imshow(I2, cmap='gray')
plt.subplot(2,2,3)
plt.imshow(I3, cmap='gray')
plt.subplot(2,2,4)
plt.imshow(I4, cmap='gray')
plt.show()

Y1 = from_mtx_to_array(I1)
Y2 = from_mtx_to_array(I2)
Y3 = from_mtx_to_array(I3)
Y4 = from_mtx_to_array(I4)

perc=0.3
N=len(Y3)

mask_to_switch = np.random.rand(N,)>perc 
mask_to_switch = (mask_to_switch-0.5)*2

plt.figure(figsize=(5,5))
plt.imshow(from_array_to_mtx(Y3), cmap='gray')
plt.title('Selected image')
plt.show()

Y = Y3*mask_to_switch

# training
Y1 = Y1.reshape((Y1.shape[0], 1))
Y2 = Y2.reshape((Y2.shape[0], 1))
Y3 = Y3.reshape((Y3.shape[0], 1))
Y4 = Y4.reshape((Y4.shape[0], 1))
W = np.matmul(Y1, Y1.T)+np.matmul(Y2, Y2.T)+np.matmul(Y3, Y3.T)+np.matmul(Y4, Y4.T)

# find the list of neurons that can switch (L neurons)
idx_neurons_to_switch = np.where(Y*np.matmul(W, Y)<0)[0]
L=len(idx_neurons_to_switch)
print('The number of neurons to switch (L) is: ', L)
# visualize initial perturbate state
plt.figure(figsize=(5,5))
plt.imshow(from_array_to_mtx(Y), cmap='gray')
plt.title('Initial perturbated image')
plt.show()
while L > 0: # until the number of neurons to switch (L) is = 0 
    # step 0: pick one random integer from 0 to L-1
    idx = np.random.randint(L)
    
    # step 1: switch that neuron (1 neuron for each step is switched)
    Y[idx_neurons_to_switch[idx]] = -1 * Y[idx_neurons_to_switch[idx]]
    
    # step 2: as the neuron value changed, the other neurons changed their value too,
    # so we recompute the list of neurons to switch (update of the list of neurons to switch)
    idx_neurons_to_switch = np.where(Y*np.matmul(W, Y)<0)[0]
    
    L=len(idx_neurons_to_switch) # update variable L 
    
    if L % 10 == 0: # visualize output only after 25 iterations
        print('The number of neurons to switch (L) is: ', L)
        plt.figure(figsize=(5,5))
        plt.imshow(from_array_to_mtx(Y), cmap='gray')
        plt.show()


# In[5]:


# Sparse patterns
X1 = np.zeros((30,30))
X1[9:20,9]=1
X1[19,9:18]=1
X2 = np.zeros((30,30))
diag=np.arange(5, 24)
for j in np.arange(len(diag)):
    X2[diag[j],diag[j]]=1
       
X3 = np.zeros((30,30))
X3[13:16,14:20]=1

plt.figure(figsize=(5,5))
plt.subplot(2,2,1)
plt.imshow(X1, cmap='gray')
plt.subplot(2,2,2)
plt.imshow(X2, cmap='gray')
plt.subplot(2,2,3)
plt.imshow(X3, cmap='gray')
plt.show()       

Y1 = from_mtx_to_array(X1)
Y2 = from_mtx_to_array(X2)
Y3 = from_mtx_to_array(X3)

a=0.02
teta = 8  # (1 mix, 5 ok 8 ok 10 ok 20 no)

perc = 0.15
N = len(Y1)
Y = np.copy(Y1)
idx_to_switch = np.where(np.random.rand(N,)<perc) 
Y[idx_to_switch] = 1 - Y[idx_to_switch]

# training
Y1 = Y1.reshape((Y1.shape[0], 1))
Y2 = Y2.reshape((Y2.shape[0], 1))
Y3 = Y3.reshape((Y3.shape[0], 1))
W = np.matmul((Y1-a), (Y1-a).T)+np.matmul((Y2-a), (Y2-a).T)+np.matmul((Y3-a), (Y3-a).T)

# find the list of neurons that can switch (L neurons)
idx_neurons_to_switch=np.where((Y-0.5)*(np.matmul(W, Y)-teta)<0)[0] 
L=len(idx_neurons_to_switch)
print('The number of neurons to switch (L) is: ', L)

plt.figure(figsize=(5,5))
plt.imshow(from_array_to_mtx(Y), cmap='gray')
plt.title('Initial perturbated image')
plt.show()
while L > 0:
    # step 0: pick one random integer from 0 to L-1
    idx = np.random.randint(L)
    # step 1: switch that neuron (1 neuron for each step is switched)
    Y[idx_neurons_to_switch[idx]] = 1 - Y[idx_neurons_to_switch[idx]]
    
    # step 2: as the neuron value changed, the other neurons changed their value too,
    # so we recompute the list of neurons to switch (update of the list of neurons to switch)
    idx_neurons_to_switch=np.where((Y-0.5)*(np.matmul(W, Y)-teta)<0)[0] 

    L=len(idx_neurons_to_switch) # update variable L     
    if L % 25 == 0: # visualize output only after 25 iterations
        print('The number of neurons to switch (L) is: ', L)
        plt.figure(figsize=(5,5))
        plt.imshow(from_array_to_mtx(Y), cmap='gray')
        plt.show()


# In[ ]:



