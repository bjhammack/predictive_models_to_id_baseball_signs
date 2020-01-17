import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2, l1
from keras.optimizers import SGD
from keras.utils import to_categorical

class Model(object):

	def __init__(self, data):
		X, y = self._transform_data(data)

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=50)
		
		print('Train Data Shape:',self.X_train.shape)
		print('Train Labels Shape:',self.y_train.shape)

	def init_network(self, output=11):
		self.model = Sequential()

		# input layer
		self.model.add(Dense(output*16, input_shape=[self.X_train.shape[1]], activation='relu', W_regularizer=l2(0.1)))
		# hidden layers
		self.model.add(Dense(output*12, activation='relu', W_regularizer=l2(0.01)))
		self.model.add(Dense(output*8, activation='relu', W_regularizer=l2(0.001)))
		#self.model.add(Dense(output*4, activation='relu', W_regularizer=l2(0.001)))
		# output layer
		self.model.add(Dense(output, activation='softmax', W_regularizer=l2(0.001)))

		sgd = SGD(lr=0.1)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

		print(self.model.summary())

	def fit_data(self, batch=405, epochs=50, verbose=2):
		history = self.model.fit(self.X_train, self.y_train, batch_size=batch, epochs=epochs, verbose=verbose, validation_data=(self.X_test, self.y_test))

		return history

	def plot_loss(self, history):
		# Summary of loss history
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'], 'g--')
		plt.title('Model Loss Over Epochs')
		plt.ylabel('Categorical Crossentropy')
		plt.xlabel('Epoch')
		plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')
		print ("CC after final iteration: ", history.history['val_loss'][-1])
		plt.show()

	def plot_accuracy(self, history):
		fig = plt.figure(figsize=(6,4))

		# Summary of accuracy history
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'], 'g--')
		plt.title('Model Accuracy Over Epochs')
		plt.ylabel('Model Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='lower left')
		print ("Accuracy after final iteration: ", history.history['val_accuracy'][-1])
		plt.show()

	def _transform_data(self, data):
		train_data = np.array([i[1] for i in data])
		train_labels = np.array([i[2] for i in data])

		vectorizer = CountVectorizer()
		vectorizer.fit(train_data)
		X = vectorizer.transform(train_data)

		encoder = OneHotEncoder()
		encoder.fit(train_labels.reshape(-1,1))
		y = encoder.transform(train_labels.reshape(-1,1))

		return X, y