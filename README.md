# powerhour
main.py (run as python main.py) will pull in the images from the data/MNIST folder and train a multi-classification convolutional neural net to predict the image label given the 28x28 pixel source. The data/MNIST folder contains 60,000 training images and 10,000 testing images.

gui.py (run as python gui.py) will open up the user interface and allow user to 'write' in a digit using the mouse then will predict its label dynamically.  The gui.py will attempt to use the model recent model from the model/ folder so there must be at least a single model in that folder to run the GUI application. 

## Installation

Dataset for the MNIST images located at http://yann.lecun.com/exdb/mnist/

Python Modules: Keras, Theano/Tensorflow

## Example


