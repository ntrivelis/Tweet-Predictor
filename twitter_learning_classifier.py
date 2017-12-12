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
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import FeatureUnion
from twitter_classes import TweetData
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier


# http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
# https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/
# https://pythontips.com/2013/08/02/what-is-pickle-in-python/

verbose = True
write_fname = 'top10_results.txt'

class TweetLearning(object):

    def __init__(self, input_filename, output_filename):
        """
        Stores and interfaces with the tweet classifier for machine learning
        """
        # Setup debugging file.
        self.write_file = open(write_fname, 'a+')
        self.write_file.write('========================================\n')
        self.write_file.write('NEW TEST\n')

        self.my_print(write_fname)

        # Instantiate the learner.
        self.td = TweetData(pickle_filename=input_filename)
        self.num_tweets = self.td.tweet_count
        self.output_filename = output_filename
        # features = []
        # favorites = []
        # time_ratio = []
        # followers = []
        # for i in range(self.td.tweet_count):
        #     features.append(self.td.tweets[i]['text'])
        #     favorites.append(self.td.tweets[i]['favorites'])
        #     time_ratio.append(self.td.tweets[i]['time_ratio'])
        #     followers.append(self.td.tweets[i]['followers'])
        #
        # favorites = np.array(favorites).reshape(self.num_tweets, 1)
        # time_ratio = np.array(time_ratio).reshape(self.num_tweets, 1)
        # followers = np.array(followers).reshape(self.num_tweets, 1)

        features_ordered = self.td.return_feature('text')
        favorites = self.td.return_feature('favorites')
        time_ratio = self.td.return_feature('time_ratio')
        followers = self.td.return_feature('followers')
        user = self.td.return_feature('screen_name')
        user_label = np.zeros((self.num_tweets,1))

        user_unique = np.unique(user)  # get unique user names
        for i, u in enumerate(user_unique):
            indices = [j for j, x in enumerate(user) if x == u]
            user_label[indices] = i
            self.my_print(len(indices))

        targets_ordered = user_label

        # shuffle
        random_order = list(np.random.permutation(self.num_tweets))
        features = [features_ordered[i] for i in random_order]
        targets = targets_ordered[random_order]

        # Number of training data samples
        self.n = len(features)
        self.my_print("Testing on {} instances...".format(self.n))

        # perform SVD

        cv = CountVectorizer()

        pipe = Pipeline([
            ('vect', cv),
            ('tfidf', TfidfTransformer())
        ])
        fs = pipe.fit_transform(features)

        ### Random Forest ###

        clf = RandomForestClassifier()
        clf.fit(fs, targets)

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        rf_top_words = []

        feature_names = cv.get_feature_names()

        for i in xrange(100):
            rf_top_words.append(feature_names[indices[i]])

        self.my_print("RF Top Words:")
        self.my_print(rf_top_words)

        ### SVD ###
        svd = TruncatedSVD(100)
        fs_svd = svd.fit_transform(fs)
        self.my_print("SVD: ")
        self.my_print(svd)

        # res = svd.transform(np.eye(2))

        back = svd.inverse_transform(fs_svd)

        # back_sum = np.sum(back,axis=0)
        back_sum = np.sum(svd.components_,0)
        top_words_ind = np.argpartition(back_sum, -20)[-20:]
        top_words_ind_sorted = top_words_ind[np.argsort(back_sum[top_words_ind])]
        top_words = []

        featnames = cv.get_feature_names()

        for wi in top_words_ind_sorted:
            # top_words.append(cv.vocabulary_.keys()[wi])
            top_words.append(featnames[wi])


        self.my_print("Top Words: ")
        self.my_print(top_words)


        # contr_sums = np.sum()

        # Linear Regression
        # clf = LinearRegression()
        # params = [
        #     {'vect__lowercase': [True]},
        #     {'vect__stop_words': ['english']},
        #     {'tfidf__sublinear_tf': [True]},
        #     {'clf__fit_intercept': [True, False]}
        #           ]
        # self.get_result_text(features, targets, clf, params)

        #
        clf = RandomForestClassifier()
        params = [
            {'vect__lowercase': [True]},
            {'vect__stop_words': ['english']},
            {'tfidf__sublinear_tf': [True]},
                  ]
        self.get_result_text(features, targets, clf, params)


        # State Vector Regression
        clf = SVC()
        params = [
            {'vect__lowercase': [True]},
            {'vect__stop_words': ['english']},
            {'tfidf__sublinear_tf': [True]},
            {'clf__C': [1]},
            {'clf__kernel': [cosine_similarity]}
                  ]
        self.get_result_text(features, targets, clf, params)

        clf = MultinomialNB()
        params = [
            {'vect__lowercase': [True]},
            {'vect__stop_words': ['english']},
            {'tfidf__sublinear_tf': [True]}
        ]
        self.get_result_text(features, targets, clf, params)

        # Adaboost Regression
        clf = AdaBoostClassifier()
        params = [
            {'vect__lowercase': [True]},
            {'vect__stop_words': ['english']},
            {'tfidf__sublinear_tf': [True]},
            {'clf__n_estimators': [25,50,100]},
                  ]
        # params = [
        #     {'vect__lowercase': [True]},
        #     {'vect__stop_words': ['english']},
        #     {'tfidf__sublinear_tf': [True]},
        #     {'clf__n_estimators': [25, 50, 100]},
        #           ]
        self.get_result_text(features, targets, clf, params)
        self.write_file.close()

    def get_result_text(self, ff, tt, clf, params):
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
        estimator.fit(ff, tt.flatten())

        self.my_print("Making predictions...")
        self.my_print("Best estimator found by grid search: {}".format(estimator.best_estimator_))
        self.my_print("Best estimator found by grid search: {}".format(estimator.best_estimator_))
        self.my_print("Best score: {}".format(estimator.best_score_))
        self.my_print("Best parameters: {}".format(estimator.best_params_))
        self.my_print("CV Results: {}".format(estimator.cv_results_))
        self.my_print("Time: {}".format(time.time()-start))
        self.my_print("========================================")

    def my_print(self, text):
        if verbose:
            print text
        self.write_file.write(str(text)+'\n')

def main(input_filename="tweet_data/twitter_data_default", output_filename="learning_data/twitter_predictions_default"):
    """
    Trains a learning model for tweet data and attempts to make like and retweet predictions base on text content and
    number of followers (inversely scaled for age of tweet).
    """
    TweetLearning(input_filename, output_filename)
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
