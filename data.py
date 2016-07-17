import sys
from os import listdir
from os.path import isfile, join
import numpy as np

from struct import *
import time



class pixel():
	def __init__(self):
		self.value = 0 # Ranges from 0.0 to 255.0
		self.x_coord = 0 # Ranges from 0 to 27
		self.y_coord = 0 # Ranges from 0 to 27

class image():
	def __init__(self):
		self.label = "NOT YET SET" # The definition for the type of image
		self.pixels = [] # Contains 784 Pixels 

	def add_pixel_XY(self, value, x_coord, y_coord):
		# We are returning false if the image is already full
		if len(self.pixels) == 784:
			return False
		temp = pixel()
		temp.value = value
		temp.x_coord = x_coord
		temp.y_coord = y_coord
		self.pixels.append(temp)
		return True

	def add_pixel(self, value, index):
		# We are returning false if the image is already full
		if len(self.pixels) == 784:
			return False
		temp = pixel()
		y_coord = index / 27
		x_coord = index % 27
		temp.value = value
		temp.x_coord = x_coord
		temp.y_coord = y_coord
		self.pixels.append(temp)
		return True

	def output_terminal(self, threshold=20):
		# Output a representation to terminal
		if len(self.pixels) != 784:
			return False
		for y in range(28):
			line = ""
			for x in range(28):
				for pixel in self.pixels:
					if pixel.x_coord == x and pixel.y_coord == y:
						if pixel.value > threshold:
							line+=" X"
						else:
							line+="  "
			print(line)
		return True

	def get_normalized_pixel_array(self):
		temp = []
		for pixel in self.pixels:
			temp.append(pixel.value/255.0)
		return temp

# Read a single word from a file and return the decimal representation
def read_word(file, index):
	vals = []
	for i in range(4):
		file.seek(i+index)
		temp = file.read(1)
		vals.append(ord(temp))

	val = vals[0]*(16**6) + vals[1]*(16**4) + vals[2]*(16**2) + vals[3]
	return val

# Read in num_words words starting at byte_offset. This is used for
# reading in the metadata of the file at the header.
def read_words(file, byte_offset, num_words):
	words = []
	for i in range(num_words):
		val = read_word(file, byte_offset+(i*4))
		words.append(val)
	return words

# Reads in a single byte from a file at index
def read_byte(file, index):
	file.seek(index)
	temp = file.read(1)
	return ord(temp)

# Reads in num_bytes bytes starting at byte_offset
def read_bytes(file, byte_offset, num_bytes):
	vals = []
	for i in range(num_bytes):
		val = read_byte(file, byte_offset+(i))
		vals.append(val)
	return vals

# Returns the next image in the file, the byte_offset start location
# must be directly after the last image or the header.
def get_image(file, byte_offset):

	pic = image()
	pixels = read_bytes(file, byte_offset, 784)
	index = 0
	for cur_pixel in pixels:
		pic.add_pixel(cur_pixel, index)
		index += 1
	return pic

# Calls get_image for num_images times
def get_images(file, byte_offset, num_images):
	pictures = []
	for i in range(num_images):
		pic = get_image(file, byte_offset+(i*784))
		pictures.append(pic)
	return pictures

# Loads a image and label set into memory
def load_data(limited, image_file="t10k-images.idx3-ubyte", label_file="t10k-labels.idx1-ubyte"):
	directory = "data/mnist"
	# List of every filename
	files = listdir(directory)
	# List of the paths to all the files
	paths = []

	# Create a list of full file paths to the files in the mnist directory
	for file in files:
		paths.append(directory+"/"+file)

	image_filepath = ""
	label_filepath = ""


	for path in paths:
		if path.find(image_file) != -1:
			image_filepath = path
		elif path.find(label_file) != -1:
			label_filepath = path
		
	print "Reading from "+str(image_filepath)

	file1 = open(image_filepath, 'rb')
	
	magic,num_images,x_range,y_range = read_words(file1,0, 4)

	#print("Magic = "+str(magic))
	print("Image Count = "+str(num_images))
	#print("x_range = "+str(x_range))
	#print("y_range = "+str(y_range))

	t0 = time.time()
	# Get the images
	pictures = get_images(file1, 4, 1000 if limited else num_images) # Third argument should be num_images
	t1 = time.time()
	image_read_time = t1-t0
	# Now fetching the labels for the images	
	print("Reading from "+str(label_filepath))

	file2 = open(label_filepath, 'rb')

	magic,num_labels = read_words(file2, 0, 2)

	#print("Magic = "+str(magic))
	print("Label Count = "+str(1000))

	t0 = time.time()
	# Getting the first several labels
	labels = read_bytes(file2, 8, 1000 if limited else num_labels) # Third argument should be num_labels
	t1 = time.time()
	label_read_time = t1-t0
	#print(labels)
	
	for pic,label in list(zip(pictures,labels)):
		pic.label = int(label)

	return pictures

def convert_image_data(data):
	image_weights = [] # List of 28x28 numpy matrices
	image_labels = [] # List of 1x9 numpy matrices

	# Iterate through each image in the input list
	for image in data:
		# Get a 784 length array of normalized values (0 to 1)
		image_values = np.array(image.get_normalized_pixel_array())

		# Reshape the data as a 1x28x28 matrix
		image_values = image_values.reshape(1, 28, 28)
		image_weights.append(image_values)

		# Create a vector of zeroes of length 10 where the only 1 value pertains
		# the label for the current image (1 at index 0 means image is a 0)
		image_label = np.zeros(10) # Empty array of length 10
		image_label[int(image.label)] = 1.0
		image_labels.append(image_label)
	
	# Return parsed, normalized, reshapen lists
	return image_weights, image_labels

def main():
	load_data()

	

if __name__ == '__main__':
	main()