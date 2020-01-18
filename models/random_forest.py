import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
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

	def init_rf(self):
		'''
		Defines random forest model.
		'''
		self.rf = RandomForestClassifier(n_estimators=100)

	def fit_data(self):
		'''
		Fits model to training data; evaluates test data. Returns the predictions and the score of the evaluation.
		'''
		self.rf.fit(self.X_train, self.y_train)

		score = self.rf.score(self.X_test, self.y_test)

		predictions = self.rf.predict(self.X_test)

		print('Accuracy:',score)

		return predictions, score

	def plot_confusion_matrix(self, labels, predictions, score):
		'''
		Plots overall accuracy and confusion matrix of each label's predicted vs actual label.
		'''
		cm = metrics.confusion_matrix(labels, predictions)
		plt.figure(figsize=(11,11))
		sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True);
		plt.ylabel('Actual label');
		plt.xlabel('Predicted label');
		all_sample_title = 'Random Forest Accuracy Score: {0}'.format(score)
		plt.title(all_sample_title, size = 15);
		plt.show()

	def predict(self, data):
		'''
		Predicts label of dataset based on previously fitted model. Returns predictions and score.
		'''
		X, y = transformations(data=data, vector=True, ohe=False, tts=False)
		
		predictions = self.rf.predict(X)
		score = self.rf.score(X, y)

		return predictions, score