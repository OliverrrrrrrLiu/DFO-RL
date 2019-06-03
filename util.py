import numpy as np


class DataPoint(object):
    def __init__(self, x, fx, grad_fx = None):
        self.x = x
        self.fx = fx
        self.grad_fx = grad_fx

    def get(self):
        return self.x, self.fx, self.grad_fx

    def store_grad(self):
        return self.grad_fx is not None
