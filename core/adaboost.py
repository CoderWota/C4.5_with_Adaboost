import sys
sys.path.append(".")
import numpy as np
from dctree.core.basemodel import BaseModel

class AdaboostClassifier(BaseModel):
    def __init__(self,base_estimator:BaseModel,n_estimators=5,learning_rate=1.0,base_estimator_params={}):
        """
        An AdaBoost classifier \n
        Params:
        base_estimator : The base estimator from which the boosted ensemble is built
        n_estimators : The maximum number of estimators at which boosting is terminated
        learning_rate : Weight applied to each classifier at each boosting iteration
        base_estimator_params : The parameters of base estimators
        """
        self.estimator = base_estimator
        self.estimator_params = base_estimator_params
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

    def init_estimators(self):
        """
        Initialize base estimators.
        """
        estimators = []
        for _ in range(self.n_estimators):
            estimator = self.estimator()
            for key,value in self.estimator_params.items():
                setattr(estimator,key,value)
            estimators.append(estimator)
        return estimators

    def init_sample_weights(self,sample_size:int,init_weight:float=None):
        """
        Initialize the sample weights.
        """
        if init_weight is not None:
            weights = np.full(sample_size,init_weight)
        else:
            weight = 1 / sample_size
            weights = np.full(sample_size,weight)
        return weights

    def calculate_error_rate(self,estimator,X:np.ndarray,Y:np.ndarray,W:np.ndarray=None):
        """
        Calculate the error rate of base estimator
        """
        if W is not None:
            return 1 - estimator.score(X,Y,W)
        else:
            return 1 - estimator.score(X,Y)

    def score(self,X:np.ndarray,Y:np.ndarray):
        """
        Return the mean accuracy on the given test data and labels \n
        Params:
        X : Test samples
        Y : True labels for X
        """
        Y_pred = self.predict(X)
        Y_comp = (Y_pred==Y).astype(np.int8)
        sum = np.sum(Y_comp)
        return sum / Y_comp.shape[0]

    def calculate_model_coefficient(self,error_rate,n_classes,epsilon=1e-6):
        """
        Calculate the coefficient of base estimator
        """
        alpha = self.learning_rate * (np.log((1-error_rate) / (error_rate + epsilon)) +\
            np.log(n_classes-1)) #SAMME
        return alpha

    def calculate_new_weights(self,coef,Y_pred:np.ndarray,Y:np.ndarray,W:np.ndarray):
        """
        Calculate new weights
        """
        W_new = np.zeros_like(W)
        for i,w in enumerate(W):
            y_pred = Y_pred[i]
            y = Y[i]
            param = coef * int(y_pred != y)
            w_new = w * np.exp(param)
            W_new[i] = w_new
        return W_new

    def _fit(self,X:np.ndarray,Y:np.ndarray):
        sample_size = X.shape[0] 
        self.n_classes = len(np.unique(Y)) #计算Y的分类数
        n_classes = self.n_classes
        self.estimators = self.init_estimators() #初始化学习器
        self.W = self.init_sample_weights(sample_size) #初始化权重
        self.coefs = np.zeros(len(self.estimators)) #初始化模型系数
        for i,estimator in enumerate(self.estimators):
            W = self.W
            estimator.fit(X,Y,sample_weight=W)
            error = self.calculate_error_rate(estimator,X,Y,W)
            coef = self.calculate_model_coefficient(error,n_classes)
            self.coefs[i] = coef
            Y_pred = estimator.predict(X)
            self.W = self.calculate_new_weights(coef,Y_pred,Y,W)

    def _predict(self,X:np.ndarray):
        len_X = X.shape[0]
        Y_pred = np.zeros(len_X,dtype=np.int32) #初始化分类结果 
        for i,row in enumerate(X):
            x = row.reshape(1,-1)
            W = np.zeros(self.n_classes) 
            for j,estimator in enumerate(self.estimators):
                y_pred = estimator.predict(x)
                W[y_pred] += self.coefs[j]
            Y_pred[i] = np.argmax(W)
        return Y_pred
    