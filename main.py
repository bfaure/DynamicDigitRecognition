import data
import model
import os

import numpy as np

LIMITED 	= False # True if we want to restrict the data to 1000 images (debugging)
LOAD 		= False # True if we want to load a prior saved model
VISUALIZE 	= True # True if we want to view some results in the command prompt

def main():

	print "Loading data..."

	# Load all of the data
	test_images = data.load_data(LIMITED)
	train_images = data.load_data(LIMITED, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")

	# Training data consits of 60000 images and 60000 labels
	# Testing data consists of 10000 images and 10000 labels

	# Each image consits of 784 (28x28) pixels each of which contains a value from
	# 0 to 255.0 which corresponds to its darkness or lightness.

	# Each input needs to be a list of numpy arrays to be valid.
	print "Normalizing data..."
	X_train, Y_train = data.convert_image_data(train_images)
	X_test, Y_test = data.convert_image_data(test_images)

	X_train = np.array(X_train)
	Y_train = np.array(Y_train)

	X_test = np.array(X_test)
	Y_test = np.array(Y_test)

	if LOAD == False:
		print "Building the model..."
		_model = model.build()
	else:
		print "Loading the model..."
		elements = os.listdir("model")
		if len(elements) == 0:
			print "No models to load."
		else:
			_model = model.load(elements[len(elements)-1])

	print "Training the model..."
	model.train(_model, X_train, Y_train, X_test, Y_test)


	if VISUALIZE:
		model.visualize(_model, test_images)

	print "Saving the model..."
	model.save(_model)

	
if __name__ == '__main__':
	main()