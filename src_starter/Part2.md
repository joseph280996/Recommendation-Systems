---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: cs135_env
    language: python
    name: python3
---

# Loading data and train SVD for embedding extraction

```python
import pandas as pd
from surprise import Dataset, NormalPredictor, Reader, SVD, accuracy
from surprise.model_selection import cross_validate
import numpy as np


from train_valid_test_loader import load_train_valid_test_datasets

# Load the dataset in the same way as the main problem 
train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()


def tuple_to_surprise_dataset(tupl):
    """
    This function convert a subset in the tuple form to a `surprise` dataset. 
    """
    ratings_dict = {
        "userID": tupl[0],
        "itemID": tupl[1],
        "rating": tupl[2],
    }

    df = pd.DataFrame(ratings_dict)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    dataset = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    return dataset

## Below we train an SVD model and get its vectors 

# train an SVD model using the training set
trainset = tuple_to_surprise_dataset(train_tuple).build_full_trainset()
algo = SVD()
algo.fit(trainset)


# Use an example to show to to slice out user and item vectors learned by the SVD 
uid = valid_tuple[0][0]
iid = valid_tuple[1][0]
rui = valid_tuple[2][0]

# Get model parameters
# NOTE: the SVD model has its own index system because the storage using raw user and item ids
# is not efficient. We need to convert raw ids to internal ids. Please read the few lines below
# carefully 

mu = algo.trainset.global_mean # SVD does not even fit mu -- it directly use the rating mean 
bu = algo.bu[trainset.to_inner_uid(uid)]
bi = algo.bi[trainset.to_inner_iid(iid)] 
pu = algo.pu[trainset.to_inner_uid(uid)] 
qi = algo.qi[trainset.to_inner_iid(iid)]

# Sanity check: we compute our own prediction and compare it against the model's prediction 
# our prediction
my_est = mu + bu + bi + np.dot(pu, qi) 

# the model's prediction
# NOTE: the training of the SVD model is random, so the prediction can be different with 
# different runs -- this is normal.   
svd_pred = algo.predict(uid, iid, r_ui=rui)

# The two predictions should be the same
print("My prediction: " + str(my_est) + ", SVD's prediction: " + str(svd_pred.est) + ", difference: " + str(np.abs(my_est - svd_pred.est)))

assert(np.abs(my_est - svd_pred.est) < 1e-6)
```

# Load in user and movies info

```python
user_info_df = pd.read_csv('../data_movie_lens_100k/user_info.csv')
movie_info_df = pd.read_csv('../data_movie_lens_100k/movie_info.csv')
```

# Data Clean Up and Feature extraction

```python
feature_vector = []
```

## Constructing Movies Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming you have loaded the movie information into a DataFrame called 'movies_df'
movie_titles = movie_info_df['title'].tolist()

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the movie titles
tfidf_matrix = tfidf_vectorizer.fit_transform(movie_titles)

# Convert the TF-IDF matrix to a dense numpy array
tfidf_features = tfidf_matrix.toarray()

# Combine the TF-IDF features with other movie features
movie_features = np.hstack((tfidf_features, movie_info_df[['release_year']].values))
```

```python
movie_features
```

## Constructing User Features

```python
user_features = user_info_df[["age", "is_male"]].values
```

```python
# Get the user and item indices from the validation tuple
user_tr = train_tuple[0]
item_tr = train_tuple[1]

# Initialize the combined feature matrix with zeros
n_user_item_pairs = len(user_tr)
n_user_features = user_features.shape[1]
n_movie_features = movie_features.shape[1]
n_user_embeddings = pu.shape[0]
n_item_embeddings = qi.shape[0]
X = np.zeros((n_user_item_pairs, n_user_features + n_movie_features + n_user_embeddings + n_item_embeddings))
print(X.shape)

# Combine user information, movie information, and embeddings
for i, (user_id, item_id) in enumerate(zip(user_tr, item_tr)):
    user_inner_id = trainset.to_inner_uid(user_id)
    item_inner_id = trainset.to_inner_iid(item_id)
    if  user_inner_id >= 0 and user_inner_id < algo.pu.shape[0]:
        X[i, -n_user_embeddings-n_item_embeddings:-n_item_embeddings] = algo.pu[user_inner_id]
    if item_inner_id >=0 and item_inner_id < algo.qi.shape[0]:
        X[i, -n_item_embeddings:] = algo.qi[item_inner_id]
    X[i, :n_user_features] = user_features[user_id]
    X[i, n_user_features:n_user_features+n_movie_features] = movie_features[item_id]

```

```python
print(X.shape)
```

# Create Binary Labeling
We'll be using a 4.5 threshold on the ratings for classification

```python
# Create target vector based on actual ratings
y = np.array(train_tuple[2])

y_binary = (y >= 4.5).astype(int)  # Binarize ratings: >= 4 is positive, < 4 is negative
print(y_binary.shape)
```

# Model Selection

```python
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = [
    ('Logistic Regression', LogisticRegression(), {'C': [0.1, 1, 10]}),
    ('SVM', SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]})
]

# Create a KFold object for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create a scorer object for AUC
auc_scorer = make_scorer(roc_auc_score)

# Perform model selection and hyperparameter tuning
best_model = None
best_params = None
best_auc = 0

for name, model, params in models:
    print(f"Evaluating {name}:")
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, params, scoring=auc_scorer, cv=kf, n_jobs=-1)
    grid_search.fit(X, y_binary)
    
    # Get the best model and its corresponding hyperparameters
    if grid_search.best_score_ > best_auc:
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_auc = grid_search.best_score_
    
    print(f"Best AUC: {grid_search.best_score_:.4f}")
    print(f"Best hyperparameters: {grid_search.best_params_}\n")

print("Best model:", best_model)
print("Best hyperparameters:", best_params)
print(f"Best AUC: {best_auc:.4f}")
```
