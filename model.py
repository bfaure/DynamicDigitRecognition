from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import model_from_json
from os import listdir

import time
import os


batch_size = 128
nb_classes = 10
nb_epoch = 12
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


def build():
	# Initializing the model
	model = Sequential()

	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
              	optimizer='adadelta',
              	metrics=['accuracy'])	

	return model

def train(model, X_train, Y_train, X_test, Y_test):

	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          	verbose=1, validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

def save(model):
	print("Saving model...")
	t = int(time.time())
	dir = "model"
	os.makedirs(dir+'/%s' %t)
	open(dir + '/%s/meta.json' % t, 'w').write(model.to_json())
	model.save_weights(dir + '/%s/data.h5' % t, overwrite=True)


def load():
	dir = "model"
	
	model = model_from_json(open(dir + '/%s/meta.json' % name).read())
	model.load_weights(dir + '/%s/data.h5' % name)
	model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
	return model