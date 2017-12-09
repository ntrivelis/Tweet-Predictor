# Attempts to predict followers for tweet data

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import time
# functions for ROC curve
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import FeatureUnion

# http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
# https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/

verbose = True
write_fname = 'Output.txt'

class twitter_learning:
    def __init__(self):
        self.write_file = open(write_fname, 'w+')
        self.write_file.write('========================================\n')
        self.write_file.write('NEW TEST\n')

        cats = ['alt.atheism', 'sci.space']
        twenty_train = fetch_20newsgroups(subset='train', categories=cats, shuffle=True, random_state=42)
        features = twenty_train.data
        targets = twenty_train.target
        # features = np.arange(200).reshape((100, 2))
        # features: list of filenames or strings
        # features = ['dog cat mouse', 'mouse dog', 'doggo dog', 'mean man bye', 'milkshake thermos']
        # targets = np.arange(5).reshape((5, 1))

        # Number of training data samples
        self.n = len(features)
        self.my_print("Testing on {} instances...".format(self.n))

        # Linear Regression
        clf = LinearRegression()
        params = [
            {'vect__lowercase': [True]},
            {'vect__stop_words': ['english']},
            {'tfidf__sublinear_tf': [True]},
            {'clf__fit_intercept': [True, False]}
                  ]
        self.get_result(features, targets, clf, params)

        # State Vector Regression
        clf = SVR()
        params = [
            {'vect__lowercase': [True]},
            {'vect__stop_words': ['english']},
            {'tfidf__sublinear_tf': [True]},
            {'clf__C': [0.1]},
            {'clf__epsilon': [0.1]},
            {'clf__kernel': ['rbf']}
                  ]
        # params = [
        #     {'vect__lowercase': [True]},
        #     {'vect__stop_words': ['english']},
        #     {'tfidf__sublinear_tf': [True]},
        #     {'clf__C': [0.1, 0.5, 1.0]},
        #     {'clf__epsilon': [0.1, 0.2, 0.5, 1.0]},
        #     {'clf__kernel': ['rbf', 'linear', 'poly']}
        #           ]
        self.get_result(features, targets, clf, params)

        # Adaboost Regression
        clf = AdaBoostRegressor()
        params = [
            {'vect__lowercase': [True]},
            {'vect__stop_words': ['english']},
            {'tfidf__sublinear_tf': [True]},
            {'clf__n_estimators': [50]},
                  ]
        # params = [
        #     {'vect__lowercase': [True]},
        #     {'vect__stop_words': ['english']},
        #     {'tfidf__sublinear_tf': [True]},
        #     {'clf__n_estimators': [25, 50, 100]},
        #           ]
        self.get_result(features, targets, clf, params)
        self.write_file.close()


    '''
    Inputs:
    features: n-by-d array of features
    targets: n-by-1 array of targets
    clf: classifier object
    params: dictionary of parameters to try
    '''
    def get_result(self, features, targets, clf, params):
        self.my_print("Making pipeline for classifier {}".format(clf))
        pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf',clf)
            ])

        # pipe = Pipeline([
        #     ('union', FeatureUnion(
        #         transformer_list=[
        #             ('text_pipe', Pipeline([
        #                 ('text', features),
        #                 ('vect', CountVectorizer()),
        #                 ('tfidf', TfidfTransformer()),
        #             ]))
        #         ]))
        #     ('clf',clf)
        #                 ])
        self.my_print("Training pipeline...")
        start = time.time()
        estimator = GridSearchCV(pipe, params)
        estimator.fit(features, targets)

        self.my_print("Making predictions...")
        self.my_print("Best estimator found by grid search: {}".format(estimator.best_estimator_))
        self.my_print("Best score: {}".format(estimator.best_score_))
        self.my_print("Time: {}".format(time.time()-start))

    def my_print(self, text):
        if verbose:
            print text
        self.write_file.write(text+'\n')

if __name__ == '__main__':
    twitter_learning()
