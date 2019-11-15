# -*- coding: utf-8 -*-
import numpy as np
from counterfactual import Counterfactual
from convexprogramming import ConvexProgram


class SeparatingHyperplane(Counterfactual, ConvexProgram):
    def __init__(self, w, b):
        self.w = w
        self.b = b

        self.epsilon = 0#1e-5

    def _build_constraints(self, var_x, y):
        return [y * (var_x.T @ self.w + self.b) >= self.epsilon]

    def compute_counterfactual(self, x, y, regularizer="l1"):        
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        return self.build_solve_opt(x, y, mad)


if __name__ == "__main__":
    # Import
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # Load data
    X, y = load_iris(True)
    X, y = X[(y == 0) | (y == 1), :], y[(y == 0) | (y == 1)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    y_target = 1.
    print(x_orig)
    print(model.predict([x_orig]))

    # Compute counterfactual
    cf = SeparatingHyperplane(model.coef_.reshape(-1, 1), model.intercept_)
    xcf = cf.compute_counterfactual(x_orig, y_target)
    print(xcf)
    print(model.predict([xcf]))
