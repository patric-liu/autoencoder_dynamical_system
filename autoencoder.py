import json 
import random
import sys
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import keras
from keras.layers import Input, Dense
from keras.models import Model

class Autoencoder(object):

	def __init__(self, enc_sizes, dec_sizes, latent_size):
		# encoder shape, [input - latent)
		self.enc_sizes = enc_sizes
		# decoder shape, [latent - output]
		self.dec_sizes = dec_sizes
		# creates three models with the same functional layers
		self.build_autoencoder(enc_sizes, dec_sizes, latent_size)


	def build_autoencoder(self, enc_sizes, dec_sizes, latent_size):
		# encoder_layers store the [input - latent) : [input, h1, h2.. hn]
		# decoder_layers store the (latent - output]  : [hn+1, hn+2.. hn+m, output]

		# create encoder input layer and store it
		encoder_layers = [Input(shape=(enc_sizes[0],))]
		# creates encoder hidden layers and links them
		for n, size in enumerate(enc_sizes[1:]):
			encoder_layers.append(Dense(size, activation='relu')(encoder_layers[n]))
		
		# create latent layer and link to encoder and store it
		latent_layer = Dense(latent_size, activation='relu')(encoder_layers[-1])

		# creates decoder hidden layers and does not link them
		decoder_layers_ = []
		for size in dec_sizes:
			decoder_layers_.append(Dense(size, activation='relu'))

		# links decoder hidden layers to encoder
		decoder_layers_enc = [decoder_layers_[0](latent_layer)]
		for index, layer in enumerate(decoder_layers_[1:]):
			decoder_layers_enc.append([layer][0](decoder_layers_enc[index]))
		
		# build autoencoder and encoder models
		self.autoencoder = Model(encoder_layers[0], decoder_layers_enc[-1])
		self.encoder = Model(encoder_layers[0], latent_layer)

		# rebuild decoder models
		latent_layer_placeholder = Input(shape=(latent_size,))
		# links decoder hidden layers to latent layer
		decoder_layers_lat = [decoder_layers_[0](latent_layer_placeholder)]
		for index, layer in enumerate(decoder_layers_[1:]):
			decoder_layers_lat.append([layer][0](decoder_layers_lat[index]))
		self.decoder = Model(latent_layer_placeholder,decoder_layers_lat[-1])


	def train(self, x_train, x_test, epochs=10, batch_size=100):
		# x_train is inputs
		# x_test is outputs
		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
		print("finished compiling autoencoder")
		self.autoencoder.fit(x_train, x_train,
						epochs = epochs,
						batch_size = batch_size,
						shuffle=True,
						validation_data=(x_test, x_test),
						verbose = 2
						)

	def predict(self, input_img):
		return self.autoencoder.predict(input_img)