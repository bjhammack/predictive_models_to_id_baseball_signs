import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

class Analytics(object):

	def __init__(self, training_data=None, training_labels=None):
		if len(training_data) > 0 and len(training_labels) > 0:
			self.training_data = training_data
			self.training_labels = training_labels

			print('Data detected. You can now call analysis commands or call "help()" for details on what you can do.')
		else:
			print('Either training data or training labels were not detected. Please re-initialie with all data.')

	def data_overview(self):
		print('Data shape:',self.training_data.shape)
		print('Data labels shape:',self.training_labels.shape)

		sign_count = [len(i.[1].split(' ')) for i in self.training_data]
		print('\nMax number of signs in set:',max(sign_count))
		print('Min number of signs in set:',min(sign_count))
		print('Avg sign set length:',np.mean(sign_count))
		print('Standard deviation:',round(np.std(sign_count)))

		print('\nUnique labels:',set([i[1] for i in self.training_labels]))
		print('Label counts:', Counter([i[1] for i in self.training_labels]))

	def histogram(self):
		sns.set()
		#counts,bins = np.histogram([i[1] for i in self.training_labels])
		labels = [i[1] for i in self.training_labels]
		sns.distplot(labels,color='b')

		# plt.hist(counts,bins,density=True,facecolr='blue',alpha=0.75)

		# plt.ylabel('Counts')
		# plt.xlabel('Signs')
		# plt.title('Unique Item Count Histogram')
		# plt.grid(True)
		# plt.show()
