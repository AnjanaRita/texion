from .sklearn import Sklearn
from .torch import Torch


class BaseClassifier:
    def __new__(cls, mode, name, params=None):
        if mode == "Sklearn":
            print(f"configured to run with {mode}")
            return Sklearn(name, params)
        if mode == "Torch":
            raise NotImplementedError("work in progress")
