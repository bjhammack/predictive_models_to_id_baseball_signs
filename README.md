# Predicting Third-Base Coach Signs with Neural Networks
Baseball is well-known for the signs and signals exchanged between players and coaches throughout the course of a ballgame. A tap of the nose, a swipe of the arm; these could be precise tactical directions for a player or a red herring designed to throw the opposing team off the scent of genuine signs.

After randomly generating a train/test dataset of genuine and red herring signs, this model will aim to classify them as accurately as possible.

# The Data
As mentioned above, the data being used for this model is randomly generated. It does not involve movement recognition or video analysis, it instead generates a 'ruleset' for what a genuine sign will be, and then creates an array of 'codes' which identify a series of signs, some genuine, others red herrings.

The code for a single sign consists of four alphbetical characters. The first indicates horizontal location of the sign (left, right, middle), the second indicates vertical location of the sign (upper, lower, middle), the third letter indicates the action being taken (tap, point, etc.), and the final letter indicates the bodypart that all these locations and actions apply to (nose, hat, etc.). An example of this code would be: 'muhc', which transaltes to a horizontal-swipe to the middle-upper chest.

These sign codes are then inserted into the procedurally generated ruleset, which introduces two new pieces. A ruleset consists of a series of sign codes (as above), with the potential additions of two special characters: `#` and ` * `. `#` indicates that a series of random signs can be inserted in this location, while ` * ` indicates that a single random sign will go in that location. Combine all these elements and a sign ruleset is created! An example of this could look something like: `# muhc * * rlpe`, which would be translated to: a series of random signs, followed by the middle-upper swipe of the chest, followed by two random signs, finished with pointing at the lower part of the right ear. This sign rule is then assigned a name from a list of potential signs (steal, bunt, etc.) and a full sign rule is created.

An array of signs is then created, some random and nonsensical, and others taken from one of the sign rules--with the special characters transformed to random signs, per the rules--until a large dataset of real and fake signs is created for the model.
