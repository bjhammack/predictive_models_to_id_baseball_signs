import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
from models._transform_data import transformations

class Model(object):

	def __init__(self, data):
		'''
		Creates train/test data for future use.
		'''
		self.X_train, self.X_test, self.y_train, self.y_test = transformations(data=data, vector=True, ohe=False, tts=True)
		
		print('Train Data Shape:',self.X_train.shape)
		print('Train Labels Shape:',self.y_train.shape)

	def init_lr(self):
		'''
		Defines logistic regression model.
		'''
		self.lr = LogisticRegression(solver='lbfgs', multi_class='auto')

	def fit_data(self):
		'''
		Fits model to training data; evaluates test data. Returns the predictions and the score of the evaluation.
		'''
		self.lr.fit(self.X_train, self.y_train)

		score = self.lr.score(self.X_test, self.y_test)

		predictions = self.lr.predict(self.X_test)

		print('Accuracy:',score)

		return predictions, score

	def plot_confusion_matrix(self, labels, predictions, score):
		'''
		Plots overall accuracy and confusion matrix of each label's predicted vs actual label.
		'''
		cm = metrics.confusion_matrix(labels, predictions)
		plt.figure(figsize=(11,11))
		sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
		plt.ylabel('Actual label');
		plt.xlabel('Predicted label');
		all_sample_title = 'Log Reg Accuracy Score: {0}'.format(score)
		plt.title(all_sample_title, size = 15);
		plt.show()

	def predict(self, data):
		'''
		Predicts label of dataset based on previously fitted model. Returns predictions and score.
		'''
		X, y = transformations(data=data, vector=True, ohe=False, tts=False)
		
		predictions = self.lr.predict(X)
		score = self.lr.score(X, y)

		return predictions, score