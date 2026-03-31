#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # Hetero-associative networks
# 
# Modify the previous module by considering a network with 16 inputs and 4 outputs. Consider as a possible input pattern 4 bars on a 4x4 screen with different orientations. Each pixel of the screen can assume values Â± 1 (first build the 4x4 matrices that represent the 4 bars and display them as images). Transform the 4x4 matrix into a vector by writing a dedicated function. 
# 
# Consider a network similar to the one used in the previous module, but with 4 output neurons, and train it with Hebb's rule and with 4 input bars.
# 
# Consider a bar corrupted by noise as an input to the network (it is advisable to add noise to the vector). To display the input on a 4x4 screen, write a function that transforms a vector into a matrix and display it as an image.
# * Analyze the ability of the network to "recognize" the bar as the variance of the noise and the slope of the output sigmoid vary. 
# * Plot the "reconstructed" bar in a final figure, starting from the exit of the network.
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
    print(nrows)
    I = np.zeros((nrows, nrows))
    for i in np.arange(nrows):
        start = i*nrows
        stop = i*nrows+nrows
        I[i,:] = V[start:stop]
    return I


# In[3]:


# defining input images
I1_no_noise = -np.ones((4,4)) 
I1_no_noise[1,:] = -I1_no_noise[1,:] 
I2_no_noise = -np.ones((4,4)) 
I2_no_noise[:,1] = -I2_no_noise[:,1] 
I3_no_noise = -np.ones((4,4))+2*np.eye(4) 
I4_no_noise = np.flipud(I3_no_noise)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(I1_no_noise)
plt.subplot(2,2,2)
plt.imshow(I2_no_noise)
plt.subplot(2,2,3)
plt.imshow(I3_no_noise)
plt.subplot(2,2,4)
plt.imshow(I4_no_noise)
plt.show()

# input normalization
V1_no_noise = from_mtx_to_array(I1_no_noise)/4 
V2_no_noise = from_mtx_to_array(I2_no_noise)/4
V3_no_noise = from_mtx_to_array(I3_no_noise)/4
V4_no_noise = from_mtx_to_array(I4_no_noise)/4

all_V = np.array([V1_no_noise, V2_no_noise, V3_no_noise, V4_no_noise]).T # shape of (16,4)
Y = np.eye(4)

# network training
W = np.matmul(Y, all_V.T)  

# network output (without noise)
Y1_no_noise = np.matmul(W, V1_no_noise)
Y2_no_noise = np.matmul(W, V2_no_noise)
Y3_no_noise = np.matmul(W, V3_no_noise)
Y4_no_noise = np.matmul(W, V4_no_noise)


# In[4]:


# defining noisy input images
sigma = 0.3
I1_noise = I1_no_noise + sigma*np.random.randn(4,4)
I2_noise = I2_no_noise + sigma*np.random.randn(4,4)
I3_noise = I3_no_noise + sigma*np.random.randn(4,4)
I4_noise = I4_no_noise + sigma*np.random.randn(4,4)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(I1_noise)
plt.subplot(2,2,2)
plt.imshow(I2_noise)
plt.subplot(2,2,3)
plt.imshow(I3_noise)
plt.subplot(2,2,4)
plt.imshow(I4_noise)
plt.show()

V1_noise = from_mtx_to_array(I1_noise)
V2_noise = from_mtx_to_array(I2_noise)
V3_noise = from_mtx_to_array(I3_noise)
V4_noise = from_mtx_to_array(I4_noise)

# input normalization
V1_noise = V1_noise/np.linalg.norm(V1_noise)
V2_noise = V2_noise/np.linalg.norm(V2_noise)
V3_noise = V3_noise/np.linalg.norm(V3_noise)
V4_noise = V4_noise/np.linalg.norm(V4_noise)

# network output (with noise)
Y1_noise = np.matmul(W, V1_noise)
Y2_noise = np.matmul(W, V2_noise)
Y3_noise = np.matmul(W, V3_noise)
Y4_noise = np.matmul(W, V4_noise)


# In[5]:


# sigmoid-activated network output (with noise) - k = 20
k = 20 
Y1_sig_no_noise = 1/(1+np.exp(-k*(Y1_no_noise - 0.5)))
Y2_sig_no_noise = 1/(1+np.exp(-k*(Y2_no_noise - 0.5)))
Y3_sig_no_noise = 1/(1+np.exp(-k*(Y3_no_noise - 0.5)))
Y4_sig_no_noise = 1/(1+np.exp(-k*(Y4_no_noise - 0.5)))

Y1_sig_noise = 1./(1+np.exp(-k*(Y1_noise - 0.5)))
Y2_sig_noise = 1./(1+np.exp(-k*(Y2_noise - 0.5)))
Y3_sig_noise = 1./(1+np.exp(-k*(Y3_noise - 0.5)))
Y4_sig_noise = 1./(1+np.exp(-k*(Y4_noise - 0.5)))


# In[7]:


# visualizations
print('#'*10+' First input')
print('linear-activated without noise:', Y1_no_noise)
print('linear-activated with noise:', Y1_noise)
print('sigmoid-activated without noise:', Y1_sig_no_noise)
print('sigmoid-activated with noise:', Y1_sig_noise)


I_output = Y1_sig_noise[0]*I1_no_noise + Y1_sig_noise[1]*I2_no_noise +Y1_sig_noise[3]*I3_no_noise + Y1_sig_noise[3]*I4_no_noise #reconstructed image

plt.figure()
plt.subplot(1,2,1)
plt.imshow(I1_noise)
plt.subplot(1,2,2)
plt.imshow(I_output)
plt.show()

print('#'*10+' Second input')
print('linear-activated without noise:', Y2_no_noise)
print('linear-activated with noise:', Y2_noise)
print('sigmoid-activated without noise:', Y2_sig_no_noise)
print('sigmoid-activated with noise:', Y2_sig_noise)

I_output = Y2_sig_noise[0]*I1_no_noise + Y2_sig_noise[1]*I2_no_noise +Y2_sig_noise[2]*I3_no_noise + Y2_sig_noise[3]*I4_no_noise
plt.figure()
plt.subplot(1,2,1)
plt.imshow(I2_noise)
plt.subplot(1,2,2)
plt.imshow(I_output)
plt.show()

print('#'*10+' Third input')
print('linear-activated without noise:', Y3_no_noise)
print('linear-activated with noise:', Y3_noise)
print('sigmoid-activated without noise:', Y3_sig_no_noise)
print('sigmoid-activated with noise:', Y3_sig_noise)

I_output = Y3_sig_noise[0]*I1_no_noise + Y3_sig_noise[1]*I2_no_noise + Y3_sig_noise[2]*I3_no_noise + Y3_sig_noise[3]*I4_no_noise
plt.figure()
plt.subplot(1,2,1)
plt.imshow(I3_noise)
plt.subplot(1,2,2)
plt.imshow(I_output)
plt.show()

print('#'*10+' Fourth input')
print('linear-activated without noise:', Y4_no_noise)
print('linear-activated with noise:', Y4_noise)
print('sigmoid-activated without noise:', Y4_sig_no_noise)
print('sigmoid-activated with noise:', Y4_sig_noise)

I_output = Y4_sig_noise[0]*I1_no_noise + Y4_sig_noise[1]*I2_no_noise + Y4_sig_noise[2]*I3_no_noise + Y4_sig_noise[3]*I4_no_noise
plt.figure()
plt.subplot(1,2,1)
plt.imshow(I4_noise)
plt.subplot(1,2,2)
plt.imshow(I_output)
plt.show()


# In[ ]:





