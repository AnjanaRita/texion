"""Base pytorch based deep classifiers
"""
import torch


class BaseTorch:
    def __init__(self, name=None, params=None):
        self.name = name
        self.params = params
        if name:
            raise NotImplementedError("Work in Progress")
