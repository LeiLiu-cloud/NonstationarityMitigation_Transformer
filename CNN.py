# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:44:35 2022

@author: lei liu

CNN architecture to predict variogram range
"""

###################
# import packages #
###################

import numpy as np                                       
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import tensorflow as tf                               
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
import wandb 
from wandb.keras import WandbCallback    

# input data shape: width and height
nx =224
ny =224
##############################
######### CNN Model ##########
##############################
def CNN_model(lr=1e-6, decay = 1e-4):
    # define neural network model sequentially
    model = Sequential()  #Lei: sequential groups a linear stack of layers
    
    # Feature map 1: (50x50x1) --> (xx)
    model.add(layers.Conv2D(32, kernel_size=(3,3), strides=2, input_shape=[nx,ny,1], padding="same"))  
    model.add(layers.ReLU())
    
    # Feature map 2: (xx) --> (xx)
    model.add(layers.Conv2D(64, kernel_size=(3,3), strides=2, padding="same"))
    model.add(layers.ReLU())
    
    # Feature map 3: (xx) --> (xx)
    model.add(layers.Conv2D(128, kernel_size=(3,3), strides=2, padding="same"))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(3, 3),strides=2))    
    
    # Feature map 5: (xx) --> (xx)
    model.add(layers.Conv2D(128, kernel_size=(3,3), strides=2, padding="same"))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(3, 3),strides=2,padding='same'))
    
    model.add(layers.Flatten())  #Lei:flatten the feature map into a 1D vector
    
    # Output layer: 128 --> 10 (i.e., each node corresponds to the probability to be each class)  
    model.add(layers.Dense(1,activation = 'linear'))   # <- Softmax is an activation function for classfier, mse

    # Compile the Neural Network - define Loss and optimizer to tune the associated weights
    opt = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8, decay=decay)
    
    model.compile(loss='mse', metrics=['mae','mape'],  optimizer='adam') #'adam'-> rmsprop #optimizer='adam'
    return model

model = CNN_model()
model.summary()

### train CNN ###
nepoch = 300; batch_size = 32 
model = CNN_model()
tf.random.set_seed(1314)
es = tf.keras.callbacks.EarlyStopping(min_delta = 0.0, monitor='val_loss', patience=50)
callbacks=[WandbCallback()]
history = model.fit(X_train,  Y_train, 
                     batch_size=batch_size, epochs=nepoch, verbose=2, 
                     validation_data=(X_test, Y_test),shuffle=True,callbacks=callbacks)
