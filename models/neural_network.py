import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
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
		self.model.add(Dense(output*4, input_shape=[self.X_train.shape[1]], activation='relu', W_regularizer=l2(0.1)))
		# hidden layers
		self.model.add(Dense(output*4, activation='relu', W_regularizer=l2(0.1)))
		self.model.add(Dropout(0.3))
		self.model.add(Dense(output*3, activation='relu', W_regularizer=l2(0.01)))
		self.model.add(Dense(output*2, activation='relu', W_regularizer=l2(0.01)))
		self.model.add(Dropout(0.2))
		# output layer
		self.model.add(Dense(output, activation='sigmoid', W_regularizer=l2(0.01)))

		sgd = SGD(lr=0.1)
		self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

		print(self.model.summary())

	def fit_data(self, batch=256, epochs=100, verbose=1):
		history = self.model.fit(self.X_train, self.y_train, batch_size=batch, epochs=epochs, verbose=verbose, validation_data=(self.X_test, self.y_test), validation_freq=10)

		return history

	def plot_loss(self, history):
		# Summary of loss history
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'], 'g--')
		plt.title('Logistic Regression Model Loss')
		plt.ylabel('Mean Squared Error')
		plt.xlabel('Epoch')
		plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')
		print ("MSE after final iteration: ", history.history['val_loss'][-1])
		plt.show()

	def plot_accuracy(self, history):
		fig = plt.figure(figsize=(6,4))

		# Summary of accuracy history
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'], 'g--')
		plt.title('Logistic Regression Model Accuracy')
		plt.ylabel('Model Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='lower left')
		print ("Accuracy after final iteration: ", history.history['val_accuracy'][-1])
		plt.show()

	def _transform_data(self, data):
		train_data_list = np.array([i[1] for i in data])
		train_labels = np.array([i[2] for i in data])

		# make all signs equal length
		'''
		max_sign_set = max([len(i.split(' ')) for i in train_data_list])
		index = 0
		for i in train_data_list:
			length = len(i.split(' '))
			while length < max_sign_set:
				train_data_list[index] += ' 0'
				length += 1
			index += 1
		train_data = np.array([i.split(' ') for i in train_data_list])
		'''
		vectorizer = CountVectorizer()
		vectorizer.fit(train_data)
		X = vectorizer.transform(train_data)

		#mlb = MultiLabelBinarizer()
		#y = mlb.fit_transform(train_labels)
		#y = pd.get_dummies(train_labels).values
		y = train_labels
		
		return X, y