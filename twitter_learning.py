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

do_print = True


def main():

    features = np.arange(200).reshape((100, 2))
    targets = np.arange(100).reshape((100, 1))

    # Number of training data samples
    n, d = features.shape
    if do_print:
        print "Testing on {} instances of {} features...".format(n,d)

    clf = LinearRegression()
    params = [
        {'vect__lowercase': [True]},
        {'vect__stop_words': ['English']},
        {'tfidf__sublinear_tf', [True]},
        {'clf__fit_intercept': [True, False]}
              ]
    get_result(features, targets, clf, params)

'''
Inputs:
features: n-by-d array of features
targets: n-by-1 array of targets
clf: classifier object
params: dictionary of parameters to try
'''
def get_result(features, targets, clf, params):
    if do_print:
        print "Making pipeline for classifier ", clf
    pipe = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                          ('clf',clf),
                         ])
    if do_print:
        print "Training pipeline..."
    start = time.time()
    estimator = GridSearchCV(pipe, params)
    pipe.fit(features, targets)

    if do_print:
        print "Making predictions..."
    # predicted = clf.predict(test_data.data)
    if do_print:
        print "Best estimator found by grid search: ", estimator.best_estimator_
        print "Accuracy: ", estimator.best_score_
    print "Time: ", time.time()-start

def plot_roc_nb(X_train, y_train, X_test, y_test, cats, cat_inds, multi_clf):
    '''
    Plots ROC curve given parameters

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param multi_clf: classifier for multiclass
    :return:
    '''

    lw = 2

    y_classes = sorted(np.unique(y_train)) # get classes
    n_classes = len(y_classes)
    if do_print:
        print "Classes: ", y_classes
        print "Number of classes: ", n_classes
    y_train = label_binarize(y_train, classes=y_classes) # binarize data
    y_test = label_binarize(y_test, classes=y_classes)

    pipe = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
                             ('tfidf', TfidfTransformer(sublinear_tf=True)),
                             ('clf_one', OneVsRestClassifier(multi_clf))
                             ])
    pipe.fit(X_train,y_train)
    y_score = pipe.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()

    # Plot all ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'r', 'g'])
    for i in range(len(cat_inds)):
        cat_ind = cat_inds[i]
        plt.plot(fpr[cat_ind], tpr[cat_ind], color=next(colors), lw=lw,
                 label='Naive Bayes: {0} (area = {1:0.2f})'
                       ''.format(cats[i], roc_auc[i]), linestyle=':')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()

def plot_roc_svm(X_train, y_train, X_test, y_test, cats, cat_inds, multi_clf):
    '''
    Plots ROC curve given parameters

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param multi_clf: classifier for multiclass
    :return:
    '''

    lw = 2
    y_classes = sorted(np.unique(y_train)) # get classes
    n_classes = len(y_classes)
    if do_print:
        print "Classes: ", y_classes
        print "Number of classes: ", n_classes
    y_train = label_binarize(y_train, classes=y_classes) # binarize data
    y_test = label_binarize(y_test, classes=y_classes)

    pipe = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
                             ('tfidf', TfidfTransformer(sublinear_tf=True)),
                             ('clf_one', OneVsRestClassifier(multi_clf))
                             ])
    pipe.fit(X_train,y_train)
    y_score = pipe.decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'r', 'g'])
    for i in range(len(cat_inds)):
        cat_ind = cat_inds[i]
        plt.plot(fpr[cat_ind], tpr[cat_ind], color=next(colors), lw=lw,
                 label='SVM: {0} (area = {1:0.2f})'
                       ''.format(cats[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()