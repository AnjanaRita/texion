"""
Base sklearn based classifiers
"""

import warnings
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

warnings.simplefilter("ignore")


OPTIONS = {'RandomForestClassifier': RandomForestClassifier,
           'MultinomialNB': MultinomialNB,
           'GaussianNB': GaussianNB,
           'SVC': SVC,
           'MLPClassifier': MLPClassifier,
           'AdaBoostClassifier': AdaBoostClassifier,
           'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis,
           'GaussianProcessClassifier': GaussianProcessClassifier,
           'ExtraTreesClassifier': ExtraTreesClassifier,
           'GradientBoostingClassifier': GradientBoostingClassifier,
           'VotingClassifier': VotingClassifier
           }


class BaseSklearn:
    """Base Sklearn classifier"""

    def __new__(cls, name, params=None):

        if name not in OPTIONS.keys():
            raise NameError(
                f"please select one of these as the name: {[x for x in OPTIONS.keys()]}")

        if params:
            clf = OPTIONS.get(name)(**params)
            print(
                f"""classification model configured to use {clf.__class__.__name__} algorithm with parameters:\n{params}""")

        else:
            clf = OPTIONS.get(name)()
            print(
                f"""classification model configured to use {clf.__class__.__name__} algorithm.\nnote: running with default configuration""")
        return clf
