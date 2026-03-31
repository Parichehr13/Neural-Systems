#!/usr/bin/env python
# coding: utf-8

# # Exercises on the Hopfield model
# 
# Load the ‘imdemos.mat’ file and extract 128x128 uint8 images (images with values from 0 to 255, e.g., ‘saturn’, ‘vertigo’, ‘coins’). Visualize the images. Convert them into binary images (images with only 0 and 1, threshold images using a threshold of 128), designing an ad-hoc function (e.g., named im2bw). Then, scale images such that each element of the matrix can have a value of -1 (background) or +1 (foreground). For reasons of memory limitations, reduce the image to a LxL=64x64 matrix, by extracting only rows and columns at even positions. 
# 
# Then, simulate a binary Hopfield network trained with the Hebb rule.
# *	Store the image in the weights of a Hopfield network with N = L2 neurons. To transform the pattern matrix into a vector of dimension L2, design an ad-hoc function (e.g., named from_mtx_to_array). Simulate the behavior of the Hopfield network, starting from a corrupted pattern, with an asynchronous update of the neurons (at each step, determine a list of neurons whose output can be updated, and randomly choose a neuron from those in the above list). To visualize the patterns, it is advisable to design an ad-hoc function to transform the vector into a matrix (e.g., named from_array_to_mtx). To corrupt a pattern, choose an assigned number of neurons randomly for the memorized pattern, and change the sign.
# 
# *	Once the good behavior of the network has been verified with a single image, repeat the exercise by storing M images with the Hebb rule. Simulate the network's ability to recover distorted images, and the possible presence of spurious images. Pay attention that the stored images are not too correlated (i.e., their scalar product is low).
# 
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path


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
    print(nrows)
    I = np.zeros((nrows, nrows))
    for i in np.arange(nrows):
        start = i*nrows
        stop = i*nrows+nrows
        I[i,:] = V[start:stop]
    return I
def im2bw(I, th_value=128):
    #th_value depends on the specific data type (uint8: 0-255 values)
    I_thresholded = np.zeros_like(I)
    I_thresholded[I>=th_value] = 1
    return I_thresholded


# In[3]:


data_path = Path(__file__).resolve().with_name('imdemos.mat')
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

plt.figure()
plt.subplot(1,3,1)
plt.imshow(X1, cmap='gray')
plt.title('saturn')
plt.subplot(1,3,2)
plt.imshow(X2, cmap='gray')
plt.title('vertigo')
plt.subplot(1,3,3)
plt.imshow(X3, cmap='gray')
plt.title('coins')
plt.show()

Y1 = from_mtx_to_array(X1)
Y2 = from_mtx_to_array(X2)
Y3 = from_mtx_to_array(X3)

Y = np.copy(Y3)
perc=0.02
N=len(Y1)
# switching a small percentage of neurons
idx_to_switch = np.where(np.random.rand(N,)<perc)
Y[idx_to_switch]=-Y[idx_to_switch]

# training
Y1 = Y1.reshape((Y1.shape[0], 1))
Y2 = Y2.reshape((Y2.shape[0], 1))
Y3 = Y3.reshape((Y3.shape[0], 1))
W = np.matmul(Y1, Y1.T)+np.matmul(Y2, Y2.T)+np.matmul(Y3, Y3.T)

# find the list of neurons that can switch (L neurons)
idx_neurons_to_switch = np.where(Y*np.matmul(W, Y)<0)[0]
L=len(idx_neurons_to_switch)

# visualize initial perturbate state
plt.figure()
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
        plt.imshow(from_array_to_mtx(Y), cmap='gray')
        plt.show()


# In[ ]:



