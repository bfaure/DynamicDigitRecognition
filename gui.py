# PyQt4 modules
import PyQt4
from PyQt4 import QtGui
from PyQt4.QtCore import QThread, QRect
from PyQt4 import QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import sys
import os
import numpy as np

# Modules pertaining to the Keras implementation
#import main
import data
import model

class execution_thread(QThread):
	# Thread to handle the calculations and inferface with the Keras classification modules.
	def __init__(self):
		QThread.__init__(self)

	def run(self):
		# Logic goes here
		model_dirs = os.listdir("model") # Traverse through the models folder and get all model names
		newest = 0
		for model_dir in model_dirs:
			name_val = int(model_dir)
			if name_val > newest:
				newest = name_val

		# newest will now hold the name of the newest model in the directory
		self.cur_model = model.load(newest) # Loading in the model

	def process_data(self):
		self.emit(SIGNAL("send_update(QString)"), "Constructing Image...")
		input_image = data.image()
		input_image.construct_from_path(self.cur_data)
		input_image.output_terminal()

		'''
		image,_ = data.convert_image_data([input_image])

		image = np.array(image)

		proba = self.cur_model.predict_proba(image)

		highest_prob = 0.0
		cur_index = 0
		highest_prob_index = 0
		#print(proba)
		for probability in proba[0]:
			if probability > highest_prob:
				highest_prob = probability
				highest_prob_index = cur_index
			cur_index += 1

		self.emit(SIGNAL("send_update(QString)"), "Digit is a "+str(highest_prob_index)+" with probability of "+str(highest_prob))
		return
		'''
		



	def get_data(self, path):
		# Need to figure out the bounds of the image (the maximums in all directions)
		self.cur_data = path
		self.process_data()


class drawing_path():
	def __init__(self):
		self.x_pos = []
		self.y_pos = []
	def add_point(self, x, y):
		self.x_pos.append(x)
		self.y_pos.append(y)
	def clear_path(self):
		self.x_pos = []
		self.y_pos = []

class window(QtGui.QWidget):
	# Window to allow user to input hand written digits for the system to analyze.
	# Basic idea is I am going to create a widget to allow the user to write in a digit
	# and when the user is done the system will gather the user input and send a signal to
	# a slot in the execution_thread which will allow it to run the Keras model and send back
	# a prediction of the digit classification.
	def __init__(self, parent=None):
		super(window, self).__init__()

		self.initThread()

	def initThread(self):
		# Initializes the thread and starts its execution (loading in the model)
		self.thread = execution_thread()
		self.thread.start()
		self.initUI()

	def initUI(self):
		# Initializes the GUI
		self.setFixedHeight(600)
		self.setFixedWidth(450)
		self.setWindowTitle("Dynamic Digit Prediction")
		self.hasDrawing = False
		self.mouseHeld = False

		self.path = drawing_path()

		self.main_layout = QtGui.QVBoxLayout(self) # Main layout for the GUI

		self.rect = QRect(0, 50, 400, 400)

		#self.drawing = QtGui.QPainter(self) # Device to allow user input
		#self.drawing.mousePressEvent.connect(self.start_drawing) # User presses mouse button
		#self.drawing.mouseMoveEvent.connect(self.drawing_occured) # User moving the mouse
		#self.drawing.mouseReleaseEvent.connect(self.end_drawing) # User lets go of mouse button

		self.label = QtGui.QLabel("Click and hold the left mouse button to draw a digit (0-9)", self)
		self.label.move(5, 10)
		self.label.setFixedWidth(300)
		self.results = QtGui.QLabel("Results will appear here", self)
		self.results.move(25, 540)
		self.results.setFixedWidth(300)
		self.result_label = QtGui.QLabel("", self)
		self.result_label.move(330, 490)


		self.clear_button = QtGui.QPushButton("Clear", self)
		self.clear_button.move(330, 535)
		self.clear_button.clicked.connect(self.clear)

		QtCore.QObject.connect(self, QtCore.SIGNAL("send_data(PyQt_PyObject)"), self.thread.get_data)
		QtCore.QObject.connect(self.thread, QtCore.SIGNAL("send_update(QString)"), self.update_label)

		self.show()

	def clear(self):
		self.path.clear_path()
		self.update()

	def mousePressEvent(self, event):
		x = event.x()
		y = event.y()

		if 100 < y < 500:
			if 25 < x < 425:
				if self.hasDrawing == True:
					self.path.clear()
				self.mouseHeld = True

				position = event.pos()
				
				self.path.add_point(x,y)

				self.results.setText("Position = "+str(position))
				return
			else:
				self.results.setText("Position out of range")
				self.mouseHeld = False
				return
		self.mouseHeld = False
		self.results.setText("Position out of range")
		return

	def mouseMoveEvent(self, event):
		x = event.x()
		y = event.y()
		if 100 < y < 500:
			if 25 < x < 425:
				if self.mouseHeld == True:

					position = event.pos()
					self.path.add_point(x,y)
					self.results.setText("Position = "+str(position))
					self.update()
					return
				else:
					return
			else:
				self.results.setText("Position out of range")
		else:
			self.results.setText("Position out of range")

	def paintEvent(self, event):
		painter = QPainter()
		painter.begin(self)

		last_x = 0
		last_y = 0
		for x,y in list(zip(self.path.x_pos, self.path.y_pos)):
			if last_x == 0:
				last_x = x
				last_y = y
			else:
				painter.drawLine(last_x, last_y, x, y)
				last_x = x
				last_y = y
		#painter.drawLine(self.last_x, self.last_y, self.cur_x, self.cur_y)
		painter.end()

	def mouseReleaseEvent(self, event):
		self.mouseHeld = False
		self.results.setText("Processing Data...")
		self.emit(SIGNAL("send_data(PyQt_PyObject)"), self.path)
		self.path.clear_path()

	def update_label(self, text):
		self.results.setText(text)


def main():

	app = QtGui.QApplication(sys.argv)
	_ = window()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()