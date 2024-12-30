#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:29:53 2018

@author: viniciuscarius
"""


from keras.layers import Dense,Input
from keras.models import Model
from keras import regularizers
from keras import backend as K
from sklearn.preprocessing import minmax_scale
import numpy as np
import os 

class AutoEncoder (object):
    
    __slots__=['n_features', 'n_neurons','n_epochs', 'batch_size', 'jobs', 'shuffle', 'verbose', '__dict__']
    def __init__ (self, n_features, n_neurons=[1024, 512, 128, 32, 2], n_epochs=150, batch_size=128, jobs=-1, shuffle=True, verbose=1):
        
        self.n_neurons=n_neurons
        self.n_features=n_features
        self.__epochs=n_epochs
        self.__batch_size=batch_size
        self.__shuffle=shuffle
        self.__verbose=verbose
        self.jobs = jobs
        
        num_processors = os.sysconf("SC_NPROCESSORS_ONLN")
        
        if self.jobs <= 0 or self.jobs > num_processors:
            self.jobs = num_processors
        
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=self.jobs, inter_op_parallelism_threads=self.jobs)))
    
    def __create_Models(self):
        input_img = Input(shape=(self.n_features,))
        #ENCODE LAYER
        # "encoded" is the encoded representation of the input
        #encoded = Dense(n_samples, activation='relu')(input_img)
        #activity_regularizer=regularizers.l1(0.001)
        encoded = Dense(self.n_neurons[0], activation='elu')(input_img)
        encoded = Dense(self.n_neurons[1], activation='elu')(encoded)
        encoded = Dense(self.n_neurons[2], activation='elu')(encoded)
        encoded = Dense(self.n_neurons[3], activation='elu')(encoded)
        encoded = Dense(self.n_neurons[4], activation='linear')(encoded)
        
        #DECODED LAYER
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(self.n_neurons[3], activation='elu')(encoded)
        decoded = Dense(self.n_neurons[2], activation='elu')(decoded)
        decoded = Dense(self.n_neurons[1], activation='elu')(decoded)
        decoded = Dense(self.n_neurons[0], activation='elu')(decoded)
        decoded = Dense(self.n_features, activation='sigmoid')(decoded)
        
        self.autoencoder = Model(inputs=input_img, outputs=decoded)
        
        self.encoder = Model(inputs=input_img, outputs=encoded)
        
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.n_neurons[-1],))
        
        # retrieve the last layer of the autoencoder model
        decoder_layer0 = self.autoencoder.layers[-5]
        decoder_layer1 = self.autoencoder.layers[-4]
        decoder_layer2 = self.autoencoder.layers[-3]
        decoder_layer3 = self.autoencoder.layers[-2]
        decoder_layer4 = self.autoencoder.layers[-1]
        
        # create the decoder model
        self.decoder = Model(inputs=encoded_input, 
                        outputs=decoder_layer4(
                            decoder_layer3(
                                    decoder_layer2(
                                        decoder_layer1(
                                            decoder_layer0(encoded_input)
                                            )
                                        )
                                    )
                                )
                            )
    def fit(self, data=None):
        self.__create_Models()
        self.data=data
        self.x_train = minmax_scale(self.data, axis = 0)
        self.x_test = self.x_train.copy()
        self.autoencoder.compile(loss='mean_squared_error', optimizer='adam')
        
        #autoencoder.compile(optimizer='adam', loss='mse', metric = ['MAPE'])
        
        # FIT
        self.history = self.autoencoder.fit(self.x_train, self.x_train,
                        epochs=self.__epochs,
                        batch_size=self.__batch_size,
                        shuffle=self.__shuffle,
                        validation_data=(self.x_test, self.x_test),
                        verbose= self.__verbose
                        )
        
        self.encoded_imgs = self.encoder.predict(self.x_test)
        self.decoded_imgs = self.decoder.predict(self.encoded_imgs)
        
        self.reduced = np.asarray(self.encoded_imgs)
        
        return self