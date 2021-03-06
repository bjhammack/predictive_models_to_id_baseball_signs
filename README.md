# Predicting Third-Base Coach Signs with Various Models
(a full, concise set of directions on how to run this project yourself can be found in full_program_command_list.txt)

Baseball is well-known for the signs and signals exchanged between players and coaches throughout the course of a ballgame. A tap of the nose, a swipe of the arm; these could be precise tactical directions for a player or a red herring designed to throw the opposing team off the scent of genuine signs.

After randomly generating a train/test dataset of genuine and red herring signs, this model will aim to classify them as accurately as possible.

# The Data
As mentioned above, the data being used for this model is randomly generated. It does not involve movement recognition or video analysis, it instead generates a 'ruleset' for what a genuine sign will be, and then creates an array of 'codes' which identify a series of signs, some genuine, others red herrings.

The code for a single sign consists of four alphbetical characters. The first indicates horizontal location of the sign (left, right, middle), the second indicates vertical location of the sign (upper, lower, middle), the third letter indicates the action being taken (tap, point, etc.), and the final letter indicates the bodypart that all these locations and actions apply to (nose, hat, etc.). An example of this code would be: 'muhc', which transaltes to a horizontal-swipe to the middle-upper chest.

These sign codes are then inserted into the procedurally generated ruleset, which introduces two new pieces. A ruleset consists of a series of sign codes (as above), with the potential additions of two special characters: `#` and ` * `. `#` indicates that a series of random signs can be inserted in this location, while ` * ` indicates that a single random sign will go in that location. Combine all these elements and a sign ruleset is created! An example of this could look something like: `# muhc * * rlpe`, which would be translated to: a series of random signs, followed by the middle-upper swipe of the chest, followed by two random signs, finished with pointing at the lower part of the right ear. This sign rule is then assigned a name from a list of potential signs (steal, bunt, etc.) and a full sign rule is created.

An array of signs is then created, some random and nonsensical, and others taken from one of the sign rules--with the special characters transformed to random signs, per the rules--until a large dataset of real and fake signs is created for the model.

You can initialize the sign generating object with `sm = train_test_sign_generator.Sign_Maker()`, then create the training data with `train_data = sm.create_train_data(<size>)`, with the optional argument `<size>` to indicate how large you want your training set (default is 100,000).

Prior to any of the models being run, the data is vectorized (the strings are converted into numerical vectors), so that the mathematics underlying the models can properly execute.

# Data Analysis
To examine the data, prior to modeling and predicting, I have created a class to handle giving insights on the data you are working with. To access this class, you need to call `analysis = data_analysis.Analytics(data)`.

As of right now, this class has three functions that each give a different perspective on the data. The first, `data_overview()`, provides us with the shape of our data, the max and min lengths of each set of signs, the average length of each set of signs, the standard deviation of each set of signs, the unique labels in the dataset, and the counts of each label in the dataset.
![data_overview](https://github.com/bjhammack/predictive_models_to_id_baseball_signs/blob/master/images/data_overview.png?raw=true "Data Overview")

`count_plot()` plots a bar chart whose x-axis is the unique data labels and y-axis is the count of each occurence of that label.
![count_plot](https://github.com/bjhammack/predictive_models_to_id_baseball_signs/blob/master/images/count_plot.png?raw=true "Count Plot")

`scatter_plot()` plots a scatterplot whose x-axis is the number of signs in each individual sign set and whose y-axis is the label each item corresponds to.
![scatter_plot](https://github.com/bjhammack/predictive_models_to_id_baseball_signs/blob/master/images/scatter_plot.png?raw=true "Scatter Plot")

# Available Models
## Neural Network
The first model designed for this project is a neural network. It can be instantiated with `nn_model = neural_network.Model(train_data)`.

When first testing the neural network, it seemed that hidden layers were only causing accuracy to wane, going from .85 accuracy without hidden layers to .2 with them. After extensive testing, it was uncovered that the hidden layers were not the problem, it was the output sizes of the input and hidden layers. The output sizes were far too small to handle an input shape of roughly 400. Once output sizes were increased, the model improved drastically, lowering its loss to .48 and increasing its accuracy by .13.

To further improve the model, the output layer's activation function was changed from 'sigmoid' to a 'softmax' activation function. This helped the model improve even more, albeit not as impressively as when output sizes were increased. Below are before and after photos when the model ran without hidden layers and 'sigmoid' activation vs. with hidden layers and 'softmax'.

Pre-Changes Loss ('Logistic Regression' was a mistype):

![before loss](https://github.com/bjhammack/predictive_models_to_id_baseball_signs/blob/master/images/nn_pre_sm_loss.png?raw=true "Before Loss")

Pre-Changes Acc:

![before accuracy](https://github.com/bjhammack/predictive_models_to_id_baseball_signs/blob/master/images/nn_pre_sm_acc.png?raw=true "Before Accuracy")

Post-Changes Loss:

![after loss](https://github.com/bjhammack/predictive_models_to_id_baseball_signs/blob/master/images/nn_post_sm_loss.png?raw=true "After Loss")

Post-Changes Acc:

![after accuracy](https://github.com/bjhammack/predictive_models_to_id_baseball_signs/blob/master/images/nn_post_sm_acc.png?raw=true "After Accuracy")

Due to the lack of features and complexity, a neural network is one of the less optimal models to use for this project; though we can improve its runtime by adjusting epochs and batch sizes based on the results we are getting. As you will see, other models easily match and outpace the neural network, while running much more quickly. A version of this project that may be more conducive to neural networks would use video of a third-base coach giving signs, rather than just a string of coded signs.

## Logistic Regression
The second model designed to predict the signs being given is logistic regression. The version of logistic regression being used is the sklearn standard LogisticRegression class.

Logistic regression performed exceptionally well on this problem, providing no lower than 0.98 and sometimes north of 0.99 accuracy on our test sets. This is of no huge surprise, as this problem is extremely well-suited for logisitic regression.

As examination of the confusion matrix -- which can be called with `lr_model.plot_confusion_matrix(score, predictions)` -- shows us that the majority of incorrect predictions revolve around the 'none' labeled signs, meaning that the model is seeing false positives, when there are no real signs. This could mean one of two thing: either actual signs were inserted into these assumed 'none' columns -- this is possibility because the function for creating these pseudo-signs does not have a robust check for this scenario yet -- or the model doesn't identify the full sequence of real signs, only parts of them. This would ensure that the true signs are almost always identified, but opens the door for false positives like we are seeing.
![confusion matrix](https://github.com/bjhammack/predictive_models_to_id_baseball_signs/blob/master/images/lr_confusion_matrix.png?raw=true)

## Random Forest
The random forest model, whose data transformation methodology closely resembles that of the log reg model, performed just as exceptionally.

Just like the logistic regression model, the random forest never rated below .98 accuracy, while consistently hitting .99 too. To reiterate the previous section, this sort of problem is especially suited for logisitic regression and random forest models, so combining the high quality of the data with that fact and you're bound to achieve impressive results.

One potentially interesting aspect of the random forest model is the confusion matrix. Just like logistic regression, most false positives involved the 'none' label, but unlike log reg it seemed specific labels had a noticeably higher rate of false positive than others. It may just be a coincidence that the test runs produced these results consistently, but further digging may reveal that the random forest structure interprets the data in the such a way that makes certain types of signs less consistent.
![confusion matrix](https://github.com/bjhammack/predictive_models_to_id_baseball_signs/blob/master/images/rf_confusion_matrix.png?raw=true)

# Thoughts on Future Iterations
One of the key things this project lacks, which I would like to add -- assuming I have the time to continue on this -- is uglier data. At first that sounds counterintuitive, but the golden rule of data science is: 'your models are only as good as the data you put in them'. This project's data is of very high quality, so it is of no surprise that the models perform so well. In the future, if I am able to return to this project, my first goal would be to incorporate noisier data; finding some ways to make everything less straight-forward. Then I would skew the data; no third-base coach signals bunt just as many times as steal. All this would be so that the models actually have to sweat to churn out good results and so that the data cleaning process could become much more robust and interesting.