import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2, l1
from keras.optimizers import SGD
from models._transform_data import transformations

class Model(object):

	def __init__(self, data):
		'''
		Creates train/test data for future use.
		'''
		self.X_train, self.X_test, self.y_train, self.y_test = transformations(data=data, vector=True, ohe=True, tts=True)
		
		print('Train Data Shape:',self.X_train.shape)
		print('Train Labels Shape:',self.y_train.shape)

	def init_network(self, output=11):
		'''
		Initializes networkl. Creates sequential model, adds layers, and compiles network for fitting.
		'''
		self.model = Sequential()

		# input layer
		self.model.add(Dense(output*16, input_shape=[self.X_train.shape[1]], activation='relu', W_regularizer=l2(0.1)))
		# hidden layers
		self.model.add(Dense(output*12, activation='relu', W_regularizer=l2(0.01)))
		self.model.add(Dense(output*8, activation='relu', W_regularizer=l2(0.001)))
		# output layer
		self.model.add(Dense(output, activation='softmax', W_regularizer=l2(0.001)))

		sgd = SGD(lr=0.1)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

		print(self.model.summary())

	def fit_data(self, batch=405, epochs=50, verbose=0):
		'''
		Fits data to model. Returns history of model.
		'''
		history = self.model.fit(self.X_train, self.y_train, batch_size=batch, epochs=epochs, verbose=verbose, validation_data=(self.X_test, self.y_test))

		return history

	def plot_loss(self, history):
		'''
		Plots loss of model based on test data, over each epoch.
		'''
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'], 'g--')
		plt.title('Model Loss Over Epochs')
		plt.ylabel('Categorical Crossentropy')
		plt.xlabel('Epoch')
		plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')
		print ("CC after final iteration: ", history.history['val_loss'][-1])
		plt.show()

	def plot_accuracy(self, history):
		'''
		Plots loss of model based on test data, over each epoch.
		'''
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'], 'g--')
		plt.title('Model Accuracy Over Epochs')
		plt.ylabel('Model Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='lower left')
		print ("Accuracy after final iteration: ", history.history['val_accuracy'][-1])
		plt.show()