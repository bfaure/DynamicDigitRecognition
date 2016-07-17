from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import data


def main():

	print "Loading data..."

	# Load all of the data
	test_images = data.load_data()
	train_images = data.load_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte")

	# Training data consits of 60000 images and 60000 labels
	# Testing data consists of 10000 images and 10000 labels

	# Each image consits of 784 (28x28) pixels each of which contains a value from
	# 0 to 255.0 which corresponds to its darkness or lightness.

	X_train = []
	Y_train = []

	X_test = []
	Y_test = []

	print "Fixing the data..'"
	# Putting the data into arrays to serve as inputs and outputs
	for image in train_images:
		pixel_weights = []
		for pixel in image.pixels:
			pixel_weights.append(pixel.value)

		X_train.append(pixel_weights)
		Y_train.append(image.label)

	for image in test_images:
		pixel_weights = []
		for pixel in image.pixels:
			pixel_weights.append(pixel.value)

		X_test.append(pixel_weights)
		Y_test.append(image.label)

	print "Building the model..."
	# Initializing the model
	model = Sequential()

	# Adding layers to the model
	model.add(Dense(64, input_dim=784, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(10, init='uniform'))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
              	optimizer=sgd,
              	metrics=['accuracy'])

	model.fit(X_train, Y_train,
          	nb_epoch=20,
          	batch_size=16)
	score = model.evaluate(X_test, Y_test, batch_size=16)


	'''
	# Compile configures the learning process of the model
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

	# Training the model
	model.fit(X_train, Y_train, nb_epoch=5, batch_size=500)

	# Used to check how the training went
	loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=50)

	# Predict new values
	classes = model.predict_classes(X_test, batch_size=50)

	# Predict new value probabilities
	proba = model.predict_proba(X_test, batch_size=50)
	'''

	



if __name__ == '__main__':
	main()