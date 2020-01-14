import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics

class Model(object):

	def __init__(self, data):
		X, y = self._transform_data(data)

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=50)
		
		print('Train Data Shape:',self.X_train.shape)
		print('Train Labels Shape:',self.y_train.shape)

	def init_lr(self):
		self.lr = LogisticRegression(solver='lbfgs', multi_class='auto')

	def fit_data(self):
		self.lr.fit(self.X_train, self.y_train)

		score = self.lr.score(self.X_test, self.y_test)

		predictions = self.lr.predict(self.X_test)

		return score, predictions

	def plot_confusion_matrix(self, score, predictions):
		# Plot overall accuracy and accuracy of each individual label
		cm = metrics.confusion_matrix(self.y_test, predictions)
		plt.figure(figsize=(11,11))
		sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
		plt.ylabel('Actual label');
		plt.xlabel('Predicted label');
		all_sample_title = 'Accuracy Score: {0}'.format(score)
		plt.title(all_sample_title, size = 15);
		plt.show()
		
	def _transform_data(self, data):
		train_data = np.array([i[1] for i in data])
		train_labels = np.array([i[2] for i in data])

		'''
		# make all signs equal length
		max_sign_set = max([len(i.split(' ')) for i in train_data_list])
		index = 0
		for i in train_data_list:
			length = len(i.split(' '))
			while length < max_sign_set:
				train_data_list[index] += ' 0'
				length += 1
			index += 1
		#train_data = np.array([i.split(' ') for i in train_data_list])
		'''

		vectorizer = CountVectorizer()
		vectorizer.fit(train_data)
		X = vectorizer.transform(train_data)

		#mlb = MultiLabelBinarizer()
		#y = mlb.fit_transform(train_labels)
		#y = pd.get_dummies(train_labels).values
		y = train_labels

		return X, y