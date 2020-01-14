import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

class Analytics(object):

	def __init__(self, data = None):
		if len(data) > 0:
			self.training_data = np.array([i[1] for i in data])
			self.training_labels = np.array([i[2] for i in data])

			print('Data detected. You can now call analysis commands or call "help()" for details on what you can do.')
		else:
			print('Data was not detected. Please re-initialize with data.')

	def data_overview(self):
		print('Data shape:',self.training_data.shape)
		print('Data labels shape:',self.training_labels.shape)

		sign_count = [len(i.split(' ')) for i in self.training_data]
		print('\nMax number of signs in set:',max(sign_count))
		print('Min number of signs in set:',min(sign_count))
		print('Avg sign set length:',np.mean(sign_count))
		print('Standard deviation:',round(np.std(sign_count)))

		print('\nUnique labels:',set(self.training_labels))
		print('\nLabel counts:', Counter(self.training_labels))

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
