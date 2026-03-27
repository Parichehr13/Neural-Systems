#!/usr/bin/env python
# coding: utf-8

# # Exercises on supervised learning: Training deep neural networks
# 
# Handwritten digit classification is a common benchmark task addressed when designing deep neural networks with gray-scale images as input. The task consists in associating to a handwritten digit the correct class, i.e., c∈{0,…,9} (10-way classification task). The most used dataset for this task is MNIST (available at http://yann.lecun.com/exdb/mnist/) and is composed by 60000 training and 10000 test images with size 28x28x1 (each pixel ij is characterized by one value, organized in a single input feature map).
# 
# See text of exercises 11 and 13. 
# 

# In[2]:


from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.signal import convolve2d
from pathlib import Path
import os
import sys

utils_dir = Path(__file__).resolve().parents[1] / "utilities_for_exercise11-12" / "Python"
if utils_dir.exists() and str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))
from load_my_digit import load_my_digit


# In[3]:


def to_one_hot(y_dense, C):
    # already provided in keras.utils.to_categorical, but ask to implement as an exercise 
    # y_dense: ndarray (n_examples,) from 0 to C-1
    n_examples = y_dense.shape[0]
    y = np.zeros((n_examples, C), dtype=int)
    for i in np.arange(n_examples):
        y[i, y_dense[i]] = 1
    return y


# In[4]:


dataset_name = 'mnist'  #or fashion_mnist', 'cifar10'
# Data from: https://keras.io/api/datasets/
# Load the data and split it between train and test sets
if dataset_name=='mnist':
    # GRAYSCALE IMAGES 28x28
    (x_train, labels_train), (x_test, labels_test) = keras.datasets.mnist.load_data()
    N = 28 # number of rows and columns of the input square matrix
    C = np.unique(labels_train).shape[0]#C = 10 # number of total classes (e.g., 10 possible digits)
elif dataset_name=='fashion_mnist':
    # GRAYSCALE IMAGES 28x28
    (x_train, labels_train), (x_test, labels_test) = keras.datasets.fashion_mnist.load_data()
    N = 28 # number of rows and columns of the input square matrix
    C = np.unique(labels_train).shape[0] #  # number of total classes (e.g., 10 possible digits)
elif dataset_name=='cifar10':
    # RGB IMAGES 32x32x3
    #Label 	Description
    #0 	airplane
    #1 	automobile
    #2 	bird
    #3 	cat
    #4 	deer
    #5 	dog
    #6 	frog
    #7 	horse
    #8 	ship
    #9 	truck
    (x_train, labels_train), (x_test, labels_test) = keras.datasets.cifar10.load_data()
    C = np.unique(labels_train).shape[0]
else:
    print('Undefined dataset')
    
print("Input data type:", x_train.dtype)
print("Min value:", x_train.min())
print("Max value:", x_train.max())
print("Shape of the training examples:", x_train.shape)
print("Shape of the test examples:", x_test.shape)

# scale images to the [0, 1] range
x_train = x_train / 255
x_test = x_test / 255
print("Input data type (after scaling):", x_train.dtype)
print("Min value (after scaling):", x_train.min())
print("Max value (after scaling):", x_train.max())

# make sure images have shape (28, 28, 1)
n_examples_train = x_train.shape[0]
n_examples_test = x_test.shape[0]
x_train = x_train.reshape((n_examples_train, N, N, 1))
x_test = x_test.reshape((n_examples_test, N, N, 1))

print("Shape of the training examples (after reshaping):", x_train.shape)
print("Shape of the test examples (after reshaping):", x_test.shape)

# convert labels to one-hot encoded labels using a custom method (to be defined in the exercises 11 and 12)
print("Shape of the trainin labels:", labels_train.shape)
print("Shape of the test labels:", labels_test.shape)
y_train = to_one_hot(labels_train, C)
y_test = to_one_hot(labels_test, C)
print("Shape of the trainin labels (after one-hot encoding):", y_train.shape)
print("Shape of the test labels (after one-hot encoding):", y_test.shape)

# Optional quick mode (keeps script runnable on limited hardware/time)
quick_examples = int(os.getenv("EX11_QUICK_TRAIN_EXAMPLES", "0"))
quick_test_examples = int(os.getenv("EX11_QUICK_TEST_EXAMPLES", "0"))
if quick_examples > 0:
    x_train = x_train[:quick_examples]
    y_train = y_train[:quick_examples]
    labels_train = labels_train[:quick_examples]
if quick_test_examples > 0:
    x_test = x_test[:quick_test_examples]
    y_test = y_test[:quick_test_examples]
    labels_test = labels_test[:quick_test_examples]


# In[5]:


# plot one example
target_idx = np.random.randint(low=0, high=x_train.shape[0], size=1)
target_img = np.squeeze(x_train[target_idx, :, :, :])
target_lbl = np.squeeze(y_train[target_idx])
plt.figure(figsize=(10,10))
plt.imshow(target_img, cmap='gray')
plt.title('One-hot label: '+ str(target_lbl))
plt.show()


# In[8]:


model_type = 'dense'  #or 'dense'
input_shape = (N, N, 1) # input dimension (grey-scale image 28 x 28, represented as a single input feature map 28x28x1)

if model_type=='dense':
    # model design
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape), # input layer
            keras.layers.Flatten(), # flatten layer (from multiple dimensions to a 1-D)

            #keras.layers.Dense(64, activation="relu"), # 1-D layer of 64 neurons, fully-connected with the preceding layer, activated via rectified linear function (ReLUs)
            #keras.layers.Dropout(0.25), # application of dropout to the output of the preceding layer with a dropout probability of 0.25 (25% neurons dropped out during training)

            keras.layers.Dense(128, activation="relu"), # 1-D layer of 128 neurons, fully-connected with the preceding layer, activated via rectified linear function (ReLUs)
            keras.layers.Dropout(0.25), # application of dropout to the output of the preceding layer with a dropout probability of 0.25 (25% neurons dropped out during training)

            keras.layers.Dense(64, activation="relu"), # 1-D layer of 128 neurons, fully-connected with the preceding layer, activated via rectified linear function (ReLUs)
            keras.layers.Dropout(0.25), # application of dropout to the output of the preceding layer with a dropout probability of 0.25 (25% neurons dropped out during training)

            keras.layers.Dense(C, activation="softmax"), # output layer with C neurons, fully-connected with the preceding layer, activated via softmax
        ]
    )

elif model_type=='cnn':
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            
            keras.layers.Conv2D(16, kernel_size=(5, 5)),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(32, kernel_size=(5, 5)),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Flatten(),
            
            keras.layers.Dense(C, activation="softmax"),
        ]
    )
else:
    print('Undefined model type')

# inspect model
model.summary()

# fully-connected nn: 109386 trainable parameters
# cnn: 18378 trainable parameters


# In[7]:


lr = 0.001 # learning rate
momentum = 0.9 # momentum term
# defining the optimizer (stochastic gradient descent with momentum)
optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum) # SGD with momentum

# compiling the model=defining the desired loss function to be minimized, the algorithm to use for the optimization process, and other metrics to track
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

valid_ratio = 0.1 # ratio of the overall training set to held back as validation set
# extracting the validation set as the first 10% of examples
x_valid = x_train[:round(valid_ratio*n_examples_train),:,:,:]
y_valid = y_train[:round(valid_ratio*n_examples_train),:]
labels_valid = labels_train[:round(valid_ratio*n_examples_train)]
# assigning back the training set as the remaining 90% of examples
x_train = x_train[round(valid_ratio*n_examples_train):,:,:,:]
y_train = y_train[round(valid_ratio*n_examples_train):,:]
labels_train = labels_train[round(valid_ratio*n_examples_train):]

# defining the ModelCheckpoint callback
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='best_mdl.keras', # set to 'best_mdl.keras' if you want only the best model, overall or to '{epoch:02d}-{val_loss:.5f}.keras' if you want to save the best model as the training proceed (best model over time)
    monitor='val_loss', # set to the metric that you want to track for the early stopped model (evaluated offline)
    save_best_only=True)

# start optimizing the network
batch_size = 128 # mini-batch size
max_epochs = int(os.getenv("EX11_MAX_EPOCHS", "100")) # maximum number of epochs
history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs=max_epochs, 
                    validation_data=(x_valid, y_valid),
                    callbacks=[model_checkpoint_callback])

# extracting training and validation losses, training and validation accuracies
train_loss = history.history['loss']
train_acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']


# In[ ]:


# visualizing losses and accuracies
epochs = np.arange(1, len(train_loss)+1)

plt.figure(figsize=(11,8))
plt.subplot(2,1,1)
plt.plot(epochs, train_loss, 'k')
plt.plot(epochs, valid_loss, 'r')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.subplot(2,1,2)
plt.plot(epochs, train_acc, 'k')
plt.plot(epochs, valid_acc, 'r')
plt.ylabel('accuracy')
plt.xlabel('epochs')

plt.legend(['training', 'validation'])
plt.show()


# In[ ]:


# loading best model 
model = keras.models.load_model('best_mdl.keras')

# model evaluation (on training, validation and test sets)
proba = model.predict(x_train) # g(X_test; theta_trained_best_model)
y_pred = np.argmax(proba, axis=-1)
cmtx = confusion_matrix(y_true=labels_train, y_pred=y_pred)
print("#"*10+'Training set')
print('Confusion matrix')
print(cmtx)
print("Accuracy: ", np.mean(labels_train==y_pred))

proba = model.predict(x_valid)
y_pred = np.argmax(proba, axis=-1)
cmtx = confusion_matrix(y_true=labels_valid, y_pred=y_pred)
print("#"*10+'Validation set')
print('Confusion matrix')
print(cmtx)
print("Accuracy: ", np.mean(labels_valid==y_pred))

proba = model.predict(x_test)
y_pred = np.argmax(proba, axis=-1)
cmtx = confusion_matrix(y_true=labels_test, y_pred=y_pred)
print("#"*10+'Test set')
print('Confusion matrix')
print(cmtx)
print("Accuracy: ", np.mean(labels_test==y_pred))

# cnn: 0.99 (test set)
# dense: 0.97 (test set)


# In[ ]:


my_digit = load_my_digit(str(utils_dir / 'sample_digit.jpg'))

plt.figure()
plt.imshow(my_digit, cmap='gray')

print("Input data type:", my_digit.dtype)
print("Shape of the my digit:", my_digit.shape)
print("Min value:", my_digit.min())
print("Max value:", my_digit.max())

# scale images to the [0, 1] range
my_digit = my_digit / 255
print("Input data type (after scaling):", my_digit.dtype)
print("Min value (after scaling):", my_digit.min())
print("Max value (after scaling):", my_digit.max())

# make sure images have shape (28, 28, 1)
my_digit = my_digit.reshape((1, N, N, 1))
print("Shape of the my digit (after reshaping):", my_digit.shape)

# model evaluation on my digit
proba = model.predict(my_digit)
y_pred_my_digit = np.argmax(proba, axis=-1)
print('Predicted class of my digit:', y_pred_my_digit)


# In[ ]:


if model_type=='cnn':
    # get conv. kernels (if model == 'cnn')
    layers = model.layers # list containing all trained layers
    layer = layers[0] # first fully-connected layer
    weights, biases = layer.get_weights() # accessing weights and biases
    print(layer.name, weights.shape)
    print(layer.name, biases.shape)


    plt.figure(figsize=(11,8))
    for i in np.arange(16):# 16 weights
        plt.subplot(4,4,i+1)
        plt.imshow(weights[:,:,0,i], cmap='gray')
        plt.title('kernel: '+ str(i))
    plt.tight_layout()


    idx = 0
    x_to_viz = x_train[idx,:, :, 0]
    plt.figure()
    plt.imshow(x_to_viz, cmap='gray')  
    plt.title('Example under investigation') 

    plt.figure(figsize=(11,8))
    for i in np.arange(16):
        plt.subplot(4,4,i+1)
        output = convolve2d(x_to_viz, weights[:,:,0,i])
        plt.imshow(output, cmap='gray')
        plt.title('output to kernel: '+ str(i))
    plt.tight_layout()
