import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import pandas as pd

class Analytics(object):

	def __init__(self, data = None):
		'''
		Creates dataframe object of dataset for analytical use and breaks the data and labels into their own np.arrays
		'''
		self.data_df = pd.DataFrame(data, columns=['id','sign','label'])
		self.data_df['sign_length'] = self.data_df['sign'].apply(lambda x: len(x.split(' ')))

		if len(data) > 0:
			self.training_data = np.array([i[1] for i in data])
			self.training_labels = np.array([i[2] for i in data])

			print('Data detected. You can now call analysis commands or call "help()" for details on what you can do.')
		else:
			print('Data was not detected. Please re-initialize with data.')

	def data_overview(self):
		'''
		Provides shape, max/min length, avg/std length, unique labels, and counts of labels of data.
		'''
		print('Data shape:',self.training_data.shape)
		print('Data labels shape:',self.training_labels.shape)

		sign_count = [len(i.split(' ')) for i in self.training_data]
		print('\nMax number of signs in set:',max(sign_count))
		print('Min number of signs in set:',min(sign_count))
		print('Avg sign set length:',np.mean(sign_count))
		print('Standard deviation:',round(np.std(sign_count)))

		print('\nUnique labels:',set(self.training_labels))
		print('\nLabel counts:', Counter(self.training_labels))

	def count_plot(self):
		'''
		Plots seaborn countplot that displays counts of each label on a bar chart.
		'''
		sns.set(style='darkgrid')
		sns.countplot(x=self.training_labels)
		plt.ylabel('Counts')
		plt.xlabel('Signs')
		plt.title('Unique Item Counts')
		plt.show()
	
	def scatter_plot(self):
		'''
		Plots scatterplot of the length of each sign set along with its label.
		'''
		sns.set(style='darkgrid')
		sns.scatterplot(x='sign_length', y='label', hue='label', data=self.data_df)
		plt.title('Unique Item Counts')
		plt.show()

	def help(self):
		print('''
The following are the available commands in data_analysis.py:

1. self.data_overview(). Prints an overview of the loaded data (averge sign length, etc.).
2. self.count_plot(). Plots bar chart with the x-axis is the labels and the y-axis is # of occurences.
3. self.scatter_plot(). Plots scatterplot with y-axis being the labels and x-axis being length of the sign-set.
''')
