import numpy as np

class BaseModel:
    def check_XY(self,X:np.ndarray,Y:np.ndarray):
        """
        Check the sample size of X and Y
        """
        size_X = X.shape[0]
        size_Y = Y.shape[0]
        return (size_X == size_Y)

    def fit(self,X:np.ndarray,Y:np.ndarray):
        """
        Fit this model with datas \n
        Params:
        X : The training input samples
        Y : The target values (class labels) as integers or strings
        """
        if self.check_Xdim(X):
            if self.check_XY(X,Y):
                self._fit(X,Y)
            else:
                raise ValueError("The size of X and Y is not same, please check your data")
        else:
            dims = np.ndim(X)
            raise ValueError("Expected 2D array, got {}D array instead".format(dims))

    def check_Xdim(self,X:np.ndarray):
        """
        Check the dimensions of X
        """
        return (np.ndim(X) == 2)
        
    def predict(self,X:np.ndarray):
        """
        Predict class for X \n
        Params: 
        X : The input samples
        """
        if self.check_Xdim(X):
            return self._predict(X)
        else:
            dims = np.ndim(X)
            raise ValueError("Expected 2D array, got {}D array instead".format(dims))

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