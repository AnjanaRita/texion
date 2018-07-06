from .sklearn.base_sklearn import BaseSklearn
from .torch.base_torch import BaseTorch


class Texion:
    def __new__(cls, mode, name, params=None):
        if mode == "Sklearn":
            print(f"configured to run with {mode}")
            return BaseSklearn(name, params)
        if mode == "Torch":
            raise NotImplementedError("work in progress")
