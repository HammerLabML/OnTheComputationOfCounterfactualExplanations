# -*- coding: utf-8 -*-
import numpy as np
from counterfactual import Counterfactual
from convexprogramming import ConvexProgram


class LinearRegression(Counterfactual, ConvexProgram):
    def __init__(self, w, b, epsilon=0.0):
        self.w = w
        self.b = b
        self.epsilon = epsilon

        super().__init__()
    
    def _build_constraints(self, var_x, y):
        return [var_x.T @ self.w + self.b - y <= self.epsilon, -1. * var_x.T @ self.w - self.b + y <= self.epsilon]

    def compute_counterfactual(self, x, y, regularizer="l1"):        
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        return self.build_solve_opt(x, y, mad)


class SoftmaxRegression(Counterfactual, ConvexProgram):
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.n_classes = self.W.shape[0]
        self.epsilon = 0.#1e-5

        super().__init__()
    
    def _build_constraints(self, var_x, y):
        return [var_x.T @ (self.W[i, :] - self.W[y, :]) + (self.b[i] - self.b[y]) + self.epsilon <= 0 for i in filter(lambda z: z != y, range(self.n_classes))]

    def compute_counterfactual(self, x, y, regularizer="l1"):        
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        return self.build_solve_opt(x, y, mad)

class PoissonRegression(Counterfactual, ConvexProgram):
    def __init__(self, w, b, epsilon=0.0):
        self.w = w
        self.b = b
        self.epsilon = epsilon

        super().__init__()
    
    def _build_constraints(self, var_x, y):
        return [var_x.T @ self.w + self.b - np.log(y) <= self.epsilon, -1. * var_x.T @ self.w - self.b + np.log(y) <= self.epsilon]

    def compute_counterfactual(self, x, y, regularizer="l1"):        
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        return self.build_solve_opt(x, y, mad)


class ExponentialRegression(Counterfactual, ConvexProgram):
    def __init__(self, w, b, epsilon=0.0):
        self.w = w
        self.b = b
        self.epsilon = epsilon

        super().__init__()
    
    def _build_constraints(self, var_x, y):
        return [var_x.T @ self.w + self.b + 1./y <= self.epsilon, -1. * var_x.T @ self.w - self.b - 1./y <= self.epsilon]

    def compute_counterfactual(self, x, y, regularizer="l1"):        
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        return self.build_solve_opt(x, y, mad)


if __name__ == "__main__":
    #####################
    # Linear regression #
    #####################
    # Import
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Lasso

    # Load data
    X, y = load_boston(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = Lasso()
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    y_orig_pred = model.predict([x_orig])
    print(y_orig_pred)

    # Compute counterfactual
    y_target = 25.
    cf = LinearRegression(model.coef_, model.intercept_, epsilon=0.1)
    xcf = cf.compute_counterfactual(x_orig, y_target)
    print(x_orig)
    print(xcf)
    print(model.predict([xcf]))
    
    ######################
    # Softmax regression #
    ######################
    # Import
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    
    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    y_target = 1
    print(x_orig)
    print(model.predict([x_orig]))

    # Compute counterfactual
    cf = SoftmaxRegression(model.coef_, model.intercept_)
    xcf = cf.compute_counterfactual(x_orig, y_target)
    print(xcf)
    print(model.predict([xcf]))

    ##########################
    # Exponential regression #
    ##########################
    # Import
    import statsmodels.api as sm

    # Load / Create data set
    data = sm.datasets.scotland.load()
    data.exog = sm.add_constant(data.exog, prepend=False)

    # Fit model
    glm_gamma = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
    glm_results = glm_gamma.fit()
    print(glm_results.summary())
    coef = glm_results.params  # Bias is in the last dimension

    # Select data point for explaining its prediction
    x_orig = data.exog[0,:]
    y_target = 20
    print(x_orig)
    print(glm_results.predict([x_orig]))
    print(data.endog[0])

    # Compute a counterfactual
    cf = ExponentialRegression(coef[:len(coef)-1], coef[-1])
    xcf = cf.compute_counterfactual(x_orig[:len(x_orig)-1], y_target)
    xcf = np.concatenate((xcf, np.array([1.])), axis=0)
    print(xcf)
    print(glm_results.predict(xcf))
    print(-1. / (np.dot(coef, xcf)))    # Note that statsmodels uses the inverse power function as a link function

    ######################
    # Poisson regression #
    ######################
    # Load / Create data set
    data = sm.datasets.scotland.load()
    print(data.names)
    data.exog = sm.add_constant(data.exog, prepend=False)

    # Fit model
    glm_poisson = sm.GLM(data.endog, data.exog, family=sm.families.Poisson())
    glm_results = glm_poisson.fit()
    print(glm_results.summary())
    coef = glm_results.params  # Again: Bias is in the last dimension

    # Select data point for explaining its prediction
    x_orig = data.exog[0,:]
    y_target = 20
    print(x_orig)
    print(glm_results.predict([x_orig]))
    print(data.endog[0])

    # Compute a counterfactual
    cf = PoissonRegression(coef[:len(coef)-1], coef[-1])
    xcf = cf.compute_counterfactual(x_orig[:len(x_orig)-1], y_target)
    xcf = np.concatenate((xcf, np.array([1.])), axis=0)
    print(xcf)
    print(x_orig)
    print(glm_results.predict(xcf))
