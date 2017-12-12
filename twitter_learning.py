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
from twitter_classes import TweetData
import sys

# http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
# https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/
# https://pythontips.com/2013/08/02/what-is-pickle-in-python/

verbose = True
write_fname = 'debug.txt'

class TweetLearning(object):

    def __init__(self, input_filename):
        """
        Stores and interfaces with the tweet classifier for machine learning
        """
        # Setup debugging file.
        self.write_file = open(write_fname, 'a+')
        self.write_file.write('========================================\n')
        self.write_file.write('NEW TEST\n')

        # Instantiate the learner.
        self.td = TweetData(pickle_filename=input_filename)

        # Extract relevant features for learning.
        features = self.td.tweets[:]['text']

        # Make sure to convert to double before dividing:
        targets = np.asarray(self.td.tweets[:]['retweets'])/np.asarray(self.td.tweets[:]['followers'])

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
        self.get_result_text(features, targets, clf, params)

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
        self.get_result_text(features, targets, clf, params)

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
        self.get_result_text(features, targets, clf, params)
        self.write_file.close()

    def get_result_text(self, features, targets, clf, params):
        '''
        Inputs:
        features: n-by-d array of features
        targets: n-by-1 array of targets
        clf: classifier object
        params: dictionary of parameters to try
        '''
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

def main(input_filename="twitter_data_default", output_filename="twitter_predictions_default"):
    """
    Trains a learning model for tweet data and attempts to make like and retweet predictions base on text content and
    number of followers (inversely scaled for age of tweet).
    """
    TweetLearning(input_filename)
    return 0

if __name__ == '__main__':
    # Choose arguments to pass.
    if len(sys.argv) == 3:
        return_arg = main(input_filename=sys.argv[1], output_filename=sys.argv[2])
    elif len(sys.argv) == 1:
        return_arg = main()
    else:
        sys.exit("error: incorrect number of input arguments")

    # Indicate success or failure.
    if return_arg == 0:
        print("success: tweet data learned and tested")
        sys.exit(0)
    else:
        sys.exit("error: failure")
