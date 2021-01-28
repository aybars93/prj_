
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



def get_observational_data(path_, path_t2, normalize=True, channels=3, next_n=7):
	"""
	Prepare dataset for prediction of  next_n number of target sequences
	"""
	print("Preparing Dataset .. \n")
	img_size = 40, 40
	filenames = glob.glob(path_+ "*")
	filenames_t2 = glob.glob(path_t2+ "*")

	data_X = []
	data_y1 = []
	data_y2 = []
	data_images = []

	for i in range(len(filenames)):
		if(channels == 3):
			img = cv2.imread(filenames[i])						# image with 3 channels
		else:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			# image with 1 channel
		img = cv2.resize(img, img_size)
		img = img.reshape(40,40, channels)				
		if(normalize == True):
			data_images.append(img/255)							# normalize them	
		else:
			data_images.append(img)			


	data_images_t2 = []
	for i in range(len(filenames_t2)):
		if(channels == 3):
			img = cv2.imread(filenames_t2[i])					# image with 3 channels
		else:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			# image with 1 channel
		img = cv2.resize(img, img_size)		
		img = img.reshape(40,40, channels)				
		if(normalize == True):
			data_images_t2.append(img/255)						# normalize them
		else:
			data_images_t2.append(img)		
	
	
	# data_X: sequence of observation_1 images for input
	# data_y1: sequence of observation_1 images for prediction
	# data_y2: sequence of observation_2 images for prediction
	for i in range(next_n, len(data_images)-next_n):
		data_X.append(data_images[:i][-next_n:])				# last next_n, observation_1 images
		data_y1.append(data_images[i:][:next_n])				# Prediction for next_n, observation_1 
		data_y2.append(data_images_t2[i:][:next_n] )			# Prediction for next_n, observation_2 images	

	# Convert to Numpy arrays
	data_X = np.asarray(data_X)
	data_y1 = np.asarray(data_y1)
	data_y2 = np.asarray(data_y2)

	return data_X, data_y1, data_y2































