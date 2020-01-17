# Predicting Third-Base Coach Signs with Various Models
Baseball is well-known for the signs and signals exchanged between players and coaches throughout the course of a ballgame. A tap of the nose, a swipe of the arm; these could be precise tactical directions for a player or a red herring designed to throw the opposing team off the scent of genuine signs.

After randomly generating a train/test dataset of genuine and red herring signs, this model will aim to classify them as accurately as possible.

# The Data
As mentioned above, the data being used for this model is randomly generated. It does not involve movement recognition or video analysis, it instead generates a 'ruleset' for what a genuine sign will be, and then creates an array of 'codes' which identify a series of signs, some genuine, others red herrings.

The code for a single sign consists of four alphbetical characters. The first indicates horizontal location of the sign (left, right, middle), the second indicates vertical location of the sign (upper, lower, middle), the third letter indicates the action being taken (tap, point, etc.), and the final letter indicates the bodypart that all these locations and actions apply to (nose, hat, etc.). An example of this code would be: 'muhc', which transaltes to a horizontal-swipe to the middle-upper chest.

These sign codes are then inserted into the procedurally generated ruleset, which introduces two new pieces. A ruleset consists of a series of sign codes (as above), with the potential additions of two special characters: `#` and ` * `. `#` indicates that a series of random signs can be inserted in this location, while ` * ` indicates that a single random sign will go in that location. Combine all these elements and a sign ruleset is created! An example of this could look something like: `# muhc * * rlpe`, which would be translated to: a series of random signs, followed by the middle-upper swipe of the chest, followed by two random signs, finished with pointing at the lower part of the right ear. This sign rule is then assigned a name from a list of potential signs (steal, bunt, etc.) and a full sign rule is created.

An array of signs is then created, some random and nonsensical, and others taken from one of the sign rules--with the special characters transformed to random signs, per the rules--until a large dataset of real and fake signs is created for the model.

You can initialize the sign generating object with `sm = train_test_sign_generator.Sign_Maker()`, then create the training data with `train_data = sm.create_train_data(<size>)`, with the optional argument `<size>` to indicate how large you want your training set (default is 100,000).

Prior to any of the models being run, the data is vectorized (the strings are converted into numerical vectors), so that the mathematics underlying the models can properly execute.

# Available Models
## Neural Network
The first model designed for this project is a neural network. It can be instantiated with `nn_model = neural_network.Model(train_data)`.

The neural network model utilizes only the input and output layers. Extensive testing determined that hidden layers only proved detrimental to the prediction process, with the amount of accuracy lost ranging from -0.15 to -0.7. With its current setup, the neural network consistently performs at and above 0.96 accuracy, with a mean squared error in the realm of <>.

Due to the lack of features and complexity, a neural network is one of the less optimal models to use for this project. As you will see, other models easily match and outpace the neural network, while running much more quickly. A version of this project that may be more conducive to neural networks would use video of a third-base coach giving signs, rather than just a string of coded signs.

## Logistic Regression
The second model designed to predict the signs being given is logistic regression. The version of logistic regression being used is the sklearn standard LogisticRegression class.

Logistic regression performed exceptionally well on this problem, providing no lower than 0.98 and sometimes north of 0.99 accuracy on our test sets. This is of no huge surprise, as this problem is extremely well-suited for logisitic regression.

As examination of the confusion matrix -- which can be called with `lr_model.plot_confusion_matrix(score, predictions)` -- shows us that the majority of incorrect predictions revolve around the 'none' labeled signs, meaning that the model is seeing false positives, when there are no real signs. This could mean one of two thing: either actual signs were inserted into these assumed 'none' columns -- this is possibility because the function for creating these pseudo-signs does not have a robust check for this scenario yet -- or the model doesn't identify the full sequence of real signs, only parts of them. This would ensure that the true signs are almost always identified, but opens the door for false positives like we are seeing.
![confusion matrix](https://github.com/bjhammack/nn_baseball_sign_predictor/blob/master/images/lr_confusion_matrix.png?raw=true)