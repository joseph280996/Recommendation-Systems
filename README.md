# Project Recommendation Systems

## Part 1
In this part of the project, I tried using the data from Netflix to recommend to the user.
Tried with CollabFilterOneVectorPerItem to experiment with Collaborative Filtering.

## Part 2

### Note
Later, I experiment with using SVD algorithm to extract embeddings as feature representation.
Along with other features from movie and user information, I experimented with 2 different classification models to decide whether to recommend the movie to the user or not.

The models tested are:
- LogisticRegression
- GradientBoostingClassifier
- XGBoostClassifier

### Result
Achieved 0.84 held-out error during training, 0.77 mae and 0.61 balanced accuracy on the test set
