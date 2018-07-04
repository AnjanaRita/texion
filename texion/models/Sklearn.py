from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import warnings
warnings.simplefilter("ignore")


options = {'RF': RandomForestClassifier, 'MNB': MultinomialNB,
           'GNB': GaussianNB, 'SVC': SVC,
           'MLP': MLPClassifier, 'AdaBoost': AdaBoostClassifier,
           'QDA': QuadraticDiscriminantAnalysis,
           'GPC': GaussianProcessClassifier,
           'ET': ExtraTreesClassifier}


class Sklearn:
    """Base Sklearn classifier"""

    def __new__(cls, name, params=None):
        if name not in options.keys():
            raise NameError(
                f"please select one of these as the name: {[x for x in options.keys()]}")
        if params:
            clf = options.get(name)(**params)
            print(
                f"""classification model configured to use {clf.__class__.__name__} \
                algorithm with parameters:\n{params}""")
        else:
            clf = options.get(name)()
            print(
                f"""classification model configured to use {clf.__class__.__name__} \
                algorithm.\nnote: running with default configuration""")
        return clf
