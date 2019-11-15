# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class Counterfactual(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def compute_counterfactual(self):
        raise NotImplementedError()
