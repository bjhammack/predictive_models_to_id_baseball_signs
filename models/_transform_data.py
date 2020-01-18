import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def transformations(data=None, vector=False, ohe=False, tts=False):
	'''
	Receives data and transforms it according to boolean parameters of function.
	'''
	if data.shape[0] < 1:
		print('Please specify data to transform.')
		
	else:
		train_data = np.array([i[1] for i in data])
		train_labels = np.array([i[2] for i in data])

		if vector == True:
			vectorizer = CountVectorizer()
			vectorizer.fit(train_data)
			X = vectorizer.transform(train_data)
		else:
			X = train_data

		if ohe == True:
			encoder = OneHotEncoder()
			encoder.fit(train_labels.reshape(-1,1))
			y = encoder.transform(train_labels.reshape(-1,1))
		else:
			y = train_labels

		if tts == True:
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
			
			return X_train, X_test, y_train, y_test
		else:
			return X, y