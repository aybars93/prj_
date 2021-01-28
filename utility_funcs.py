
import numpy as np
import matplotlib.pyplot as plt
import pickle




def get_images_from_numpy(path_file, colored=True, save=True):
	"""
	Obtain images from numpy data and save them
	"""
	predicted_series = np.load(path_file, allow_pickle=True) 
	print("Predicted series shape is : " + str(predicted_series.shape) )

	# rows, columns = 4, 9
	rows = 4
	columns = predicted_series.shape[0] // 4
	fig, axes = plt.subplots(rows, columns)
	counter_ = 0
	for i in range(rows):
		for j in range(columns):	
			if(colored == True):
				axes[i, j].imshow(predicted_series[counter_])		
			else:
				axes[i, j].imshow(predicted_series[counter_].reshape(40,40), cmap="gray")
				
			axes[i, j].axis('off')
			counter_ += 1
	
	if(save == True):
		plt.savefig('image_output.png')
	# plt.show()
	# plt.close()
	plt.close(fig)




def get_saved_model(path_model):
	model = keras.models.load_model(path_model)
	return model

























