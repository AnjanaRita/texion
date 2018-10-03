from .sklearn.base_sklearn import BaseSklearn
from .torch.base_torch import BaseTorch


class Texion:
    def __new__(cls, mode, name, params=None):
        """

        parameters: 
        __________

        mode: str
            configures Texion's Backend, choose either `Sklearn` or `Torch`.

        name: str
            configures the algorithm/architecture to be used for classification

        params: dict
            optional, a dictionary containing the model hyperparameters

        returns: 
        _______

            Texion text classifier


        """
        if mode == "Sklearn":
            print(f"configured to run with {mode} Backend")
            return BaseSklearn(name, params)
        if mode == "Torch":
            print(f"configured to run with {mode} Backend")
            return BaseTorch(name, params)
