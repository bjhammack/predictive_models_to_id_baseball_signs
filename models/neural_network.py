import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.regularizers import l2, l1
from keras.optimizers import SGD

class Model(object):

	def __init__(self, data):
		train_data = np.array([i[1] for i in data])
		train_labels = np.array([i[2] for i in data])

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=50)
		
		print('Train Data Shape:',self.X_train.shape)
		print('Train Labels Shape:',self.y_train.shape)

	def init_network(self, output=10, hidden_layers=3, activations=['sigmoid','relu','relu'], w_reg=[0.1,0.1,0.01]):
		self.model = Sequential()
		self.model.add(Dense(output*(hidden_layers+1), input_shape=[1], activation='sigmoid', W_regularizer=l2(0.1)))

		output_multiplier = hidden_layers
		for i in range(0,hidden_layers):
			self.model.add(Dense(output*output_multiplier, activation=activations[i], W_regularizer=l2(w_reg[i])))

			output_multiplier -= 1

		self.model.add(Dense(output, activation='sigmoid', W_regularizer=l2(0.01)))
		self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

		print(self.model.summary())

	def fit_data(self, batch=256, epochs=100, verbose=1):
		history = self.model.fit(self.X_train, self.y_train, batch_size=batch, epochs=epochs, verbose=verbose, validation_data=(self.X_test, self.y_test))

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

	def plot_accuracy(self, historoy):
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