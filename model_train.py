from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pylab as plt

import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model	# for model plotting

import utility_funcs
import data_manager



def getModel(filter_channels=64, output_channels=3):
	"""
		It is better to use normalization after any layer
		
		output_channels : 3 for colored images, 1 for gray images
	"""

	Input_1= keras.Input(shape=(None, 40, 40, 3))

	x = layers.ConvLSTM2D(filters=filter_channels, kernel_size=(3, 3),  padding="same", return_sequences=True, name="input_observation1")(Input_1)
	x = layers.BatchNormalization()(x)
	x = layers.ConvLSTM2D(filters=filter_channels, kernel_size=(3, 3),  padding="same", return_sequences=True)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ConvLSTM2D(filters=filter_channels, kernel_size=(3, 3),  padding="same", return_sequences=True)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ConvLSTM2D(filters=filter_channels, kernel_size=(3, 3),  padding="same", return_sequences=True)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ConvLSTM2D(filters=filter_channels, kernel_size=(3, 3),  padding="same", return_sequences=True)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ConvLSTM2D(filters=filter_channels, kernel_size=(3, 3),  padding="same", return_sequences=True)(x)
	x = layers.BatchNormalization()(x)

	
	out_1 = layers.Conv3D(filters=output_channels, kernel_size=(3, 3, 3), activation="sigmoid", padding="same", name="observation1")(x)				
	out_2 = layers.Conv3D(filters=output_channels, kernel_size=(3, 3, 3), activation="sigmoid", padding="same", name="observation2")(x)				

	model = Model(inputs = Input_1, outputs = [out_1, out_2] )					# multi output
	# model.compile(loss = 'binary_crossentropy', optimizer = "adam")			# adam optimizer produces better result	
	model.compile(loss = 'mse', optimizer = "adam")								# use MSE loss for regression task
	
	return model




def train_model(path_observation_1, path_observation_2, filter_channels=64, output_channels=3, epochs=3000, plot=False):
	model_ = getModel(filter_channels, output_channels)
	# plot_model(model_, to_file="_model_.png")
	
	# path_observation_1 = ""
	# path_observation_2 = ""
	data_X, data_y1, data_y2 = data_manager.get_observational_data(path_observation_1, path_observation_2, next_n=7)		# prediction of next 7 images
	data_X = tf.convert_to_tensor(data_X)
	data_y1 = tf.convert_to_tensor(data_y1)
	data_y2 = tf.convert_to_tensor(data_y2)

	# loss = []
	# val_loss = []
	class PredictionCallbackClass(tf.keras.callbacks.Callback):    
		def on_epoch_end(self, epoch, logs={}):
			test_index = 128
			test_series = data_X[test_index]				
			predicted_series = test_series[:7, ::, ::, ::]				# Start from first 7 frames

			for j in range(7):
				predicted_output = self.model.predict(predicted_series[np.newaxis, ::, ::, ::, ::])		
				predicted_first = predicted_output[0][::, -1, ::, ::, ::]								# concatenate first observation image prediction
				predicted_series = np.concatenate((predicted_series, predicted_first), axis=0)    			
				predicted_second = predicted_output[1][::, -1, ::, ::, ::]								# concatenate second observation image prediction]
				predicted_series = np.concatenate((predicted_series, predicted_second), axis=0)   	
				
			# pth_folder = ".//predicted_folder//"
			# np.save(pth_folder +  'data' + str(epoch) + '.npy', predicted_series) # save
			np.save('data' + str(epoch) + '.npy', predicted_series) # save
			utility_funcs.get_images_from_numpy('data' + str(epoch) + '.npy', colored=True, save=True)


	# list of callbacks
	list_callbacks = [
		tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor = 0.1, patience = 5),	# change hyperparameter dynamically
		# tf.keras.callbacks.EarlyStopping(monitor="loss", patience = 5),						# disable early stopping
		tf.keras.callbacks.TensorBoard(log_dir="my_tensor_log_dir", histogram_freq=1,),			# for visualization 
		tf.keras.callbacks.ModelCheckpoint(filepath="model_best.h5", monitor="val_loss", save_best_only=True),		# save best model only
		PredictionCallbackClass()
	]
		
	history = model_.fit(data_X, [data_y1, data_y2] , batch_size=8, epochs=epochs, verbose=2, validation_split=0.1, callbacks=list_callbacks)	# train and validate after each epoch
	
	# Plot in the end
	if(plot == True):		
		loss_ = history.history['loss']
		val_loss = history.history['val_loss']
		epochs_ = list(range(1, len(loss_)+1))
		plt.plot(epochs_, loss_, 'bo', label='Training Loss')
		plt.plot(epochs_, val_loss, 'b', label='Validation Loss')
		plt.title("Training and Validation Loss")
		plt.legend()
		plt.show()













