import numpy as np

class CVProcessor:
    def __init__(self,n_splits:int=1,criterion="gini"):
        """
        C4.5 based continuous variable processor, covert continuous variable to discrete variable \n
        Params:
        n_splits: Number of splits, split continuous variable max to 2^N
        criterion : The function to measure the quality of a split {"gini", "entropy"}
        """
        self.n_splits = n_splits
        self.slices = None
        if criterion == "gini":
            self.gini = True
        elif criterion == "entropy":
            self.gini = False

    def count_with_weights(self,X:np.ndarray):
        """
        Count p with weights
        """
        if 0 in X: counts = np.bincount(X)
        else: counts = np.bincount(X)[1:]
        length = X.shape[0]
        p = counts / length
        return p

    def calculate_entropy(self,X:np.ndarray,epsilon=1e-5):
        """
        Calculate entropy of X
        """
        p = self.count_with_weights(X)
        entropies = -p * np.log2(p + epsilon)
        return np.sum(entropies)

    def calculate_gini(self,X:np.ndarray):
        """
        Calculate Gini impurity of X
        """
        p = self.count_with_weights(X)
        sum_p2 = np.sum(np.power(p,2))
        return 1 - sum_p2

    def calculate_impurity(self,X:np.ndarray):
        """
        Calculate the impurity of X
        """
        if self.gini:
            return self.calculate_gini(X)
        else:
            return self.calculate_entropy(X)

    def sort(self,X:np.ndarray,Y:np.ndarray,I:np.ndarray):
        """
        Sort X,Y,indices by X
        """
        sorted_index = np.argsort(X)
        return X[sorted_index],Y[sorted_index],I[sorted_index]

    def split_data_by_idx(self,X:np.ndarray,Y:np.ndarray,I:np.ndarray,index:int):
        """
        Split Y by index to Y[:index] and Y[:index]
        Return :
        X_splitted : splitted X
        Y_splitted : splitted Y
        I_splitted : splitted I 
        """
        X_splitted = (X[:index],X[index:])
        Y_splitted = (Y[:index],Y[index:])
        I_splitted = (I[:index],I[index:])
        return X_splitted,Y_splitted,I_splitted

    def calculate_splitted_impurity(self,Y_splitted):
        length = np.sum([Y_sub.shape[0] for Y_sub in Y_splitted])
        impurity_total = 0
        for Y_sub in Y_splitted:
            counts = Y_sub.shape[0]
            impurity = self.calculate_impurity(Y_sub)
            p = counts / length
            impurity_total += p * impurity
        return impurity_total
    
    def split_data(self,X:np.ndarray,Y:np.ndarray,I:np.ndarray,depth,split_indices:list):
        if depth <= self.n_splits and X.shape[0] > 1:
            gains = np.zeros(X.shape[0])
            for idx in range(1,X.shape[0]):
                X_subs,Y_subs,I_subs = self.split_data_by_idx(X,Y,I,idx)
                Y_impurity = self.calculate_impurity(Y)
                splitted_impurity = self.calculate_splitted_impurity(Y_subs)
                gain = Y_impurity - splitted_impurity
                if gain > 0: gains[idx] = gain
            max_idx = np.argmax(gains)
            if max_idx > 0:
                index = I[max_idx]
                split_indices.append(index)
            X_subs,Y_subs,I_subs = self.split_data_by_idx(X,Y,I,max_idx)
            for idx in range(len(X_subs)):
                X_sub = X_subs[idx]
                Y_sub = Y_subs[idx]
                I_sub = I_subs[idx]
                self.split_data(X_sub,Y_sub,I_sub,depth+1,split_indices)

    def indices_to_values(self,X:np.ndarray,indices:list):
        """
        X: sorted X
        """
        values = np.zeros(len(indices))
        for i,idx in enumerate(indices):
            value = (X[idx-1] + X[idx])/2
            values[i] = value
        values = np.unique(values)
        return values

    def fit(self,X:np.array,Y:np.array):
        """
        Fit this model with datas \n
        Params:
        X : The training input samples, 1D array
        Y : The target values (class labels) as integers or strings, 1D array
        """
        split_indices = []
        I = np.array(range(len(X)))
        X,Y,I = self.sort(X,Y,I)
        self.split_data(X,Y,I,1,split_indices)
        self.slices = self.indices_to_values(X,split_indices)

    def predict(self,X:np.ndarray):
        """
        Convert X to a categorical variable
        """
        X_dummied = np.digitize(X,bins=self.slices)
        return X_dummied