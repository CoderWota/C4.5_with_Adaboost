import sys
sys.path.append(".")
import numpy as np
from dctree.core.basemodel import BaseModel

def mode(array:np.ndarray,return_percents=False):
    """
    Return an array of modal values
    """
    uniques, counts = np.unique(array, return_counts=True)
    modes = uniques[counts == np.amax(counts)]
    if return_percents:
        percents = counts / np.sum(counts)
        p = percents[counts == np.amax(counts)]
        return modes,p
    else: 
        return modes

class Node:
    def __init__(self,children:dict,depth,attribute=None,result=None,is_leaf=False,rate=None):
        """
        Node of decision tree \n
        Params:
        attribute : The attribute of this node
        children : The child nodes
        depth : The depth of this node
        (is_leaf) : Is this node a leaf node
        (result) : Output result of leaf node
        (rate) : The percentages of result
        """
        self.attr = attribute
        self.children = children
        self.depth = depth
        self.result = result
        self.is_leaf = is_leaf
        self.rate = rate
        self.prune_gain = None

class DecisionTreeClassifier(BaseModel):
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self,value):
        if value < 0:
            raise ValueError("Alpha should be bigger than zero")
        else:
            if value > 0: 
                self.prunning = True
                self._alpha = value
            else: 
                self.prunning = False

    def __init__(self,criterion:str="gini",max_depth=np.inf,alpha:float=0):
        """
        A C4.5 based decision tree classifier \n
        Params:
        criterion : The function to measure the quality of a split {"gini", "entropy"}
        max_depth : The maximum depth of the tree (pre-prunning)
        alpha : The alpha value for post-prunning, if alpha bigger than zero, then start prunning
        """
        self._depth = 0
        self.max_depth = max_depth
        self.is_weighted = False
        self.alpha = alpha
        if criterion == "gini":
            self.gini = True
        elif criterion == "entropy":
            self.gini = False
        else:
            raise ValueError("Input error: You have to choose a criterion from {\"gini\", \"entropy\"}")

    def count_with_weights(self,X:np.ndarray,W:np.ndarray):
        """
        Count p with weights
        """
        if 0 in X: counts = np.bincount(X,W)
        else: counts = np.bincount(X,W)[1:]
        length = np.sum(W)
        p = counts / length
        return p
    
    def calculate_entropy(self,X:np.ndarray,W:np.ndarray,epsilon=1e-5):
        """
        Calculate entropy of X
        """
        p = self.count_with_weights(X,W)
        entropies = -p * np.log2(p + epsilon)
        return np.sum(entropies)

    def calculate_gini(self,X:np.ndarray,W:np.ndarray):
        """
        Calculate Gini impurity of X
        """
        p = self.count_with_weights(X,W)
        sum_p2 = np.sum(np.power(p,2))
        return 1 - sum_p2

    def calculate_impurity(self,X:np.ndarray,W:np.ndarray):
        """
        Calculate the impurity of X
        """
        if self.gini:
            return self.calculate_gini(X,W)
        else:
            return self.calculate_entropy(X,W)

    def split_YW_with_Xsub(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray):
        """
        Split Y and W by X
        X: sub X, 1D array
        Y: Y, 1D array
        W: W, 1D array
        """
        Y_subs = []
        W_subs = []
        classes = np.unique(X)
        for cls in classes:
            idx = np.where(X==cls)
            Y_sub = Y[idx]
            W_sub = W[idx]
            Y_subs.append(Y_sub)
            W_subs.append(W_sub)
        return Y_subs,W_subs

    def calculate_weighted_impurity(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray): #weight?
        """
        Calculate weighted impurity
        X: sub X, 1D array
        Y: Y, 1D array
        W: W, 1D array
        """
        weights = self.count_with_weights(X,W)
        Y_subs,W_subs = self.split_YW_with_Xsub(X,Y,W)
        weighted_impurity = 0
        for i in range(len(Y_subs)):
            Y_sub = Y_subs[i]
            W_sub = W_subs[i]
            impurity = self.calculate_impurity(Y_sub,W_sub)
            weight = weights[i]
            weighted_impurity += weight * impurity
        return weighted_impurity

    def calculate_info_gain(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray):
        """
        Calculate information gain of attribute X_a
        X: sub X, 1D array
        Y: Y, 1D array
        W: W, 1D array
        """
        weighted_impurity = self.calculate_weighted_impurity(X,Y,W)
        impurity_Y = self.calculate_impurity(Y,W)
        return impurity_Y - weighted_impurity

    def calculate_info_gain_ratio(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray,epsilon=1e-6):
        """
        Calculate information gain ratio of attribute X_a
        X: sub X, 1D array
        Y: Y, 1D array
        """
        info_gain = self.calculate_info_gain(X,Y,W)
        impurity_X = self.calculate_impurity(X,W)
        return info_gain / (impurity_X + epsilon)

    def split_XYW_by_attr(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray,attr_col:int):
        """
        Split X,Y and W into subsets by attribute
        """
        X_attr = X[:,attr_col]
        classes = np.unique(X_attr)
        X_subs = []
        Y_subs = []
        W_subs = []
        for cls in classes:
            idx = np.where(X_attr==cls)
            X_sub = X[idx]
            Y_sub = Y[idx]
            W_sub = W[idx]
            X_subs.append(X_sub)
            Y_subs.append(Y_sub)
            W_subs.append(W_sub)
        return X_subs,Y_subs,W_subs

    def calculate_split_gain(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray):
        """
        Calculate the criterion for spliting
        """
        if self.gini:
            split_gain = self.calculate_info_gain(X,Y,W)
        else:
            split_gain = self.calculate_info_gain_ratio(X,Y,W)
        return split_gain
    
    def calculate_max_attribute(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray,A:set):
        """
        Calculate the attribute with biggest gain \n
        X : 2D Array
        Y : 1D Array
        A : attributes, 1D set
        """
        gains = np.zeros(len(A))
        A_array = np.array(list(A),dtype=np.int32)
        for i,a in enumerate(A_array):
            X_attr = X[:,a]
            split_gain = self.calculate_split_gain(X_attr,Y,W)
            gains[i] = split_gain
        max_idx = np.argmax(gains)
        max_gain = np.max(gains)
        return A_array[max_idx]

    def is_all_Y_in_one_class(self,Y:np.ndarray):
        """
        Judge if all data of Y is in one class
        """
        uniques = np.unique(Y)
        return (len(uniques) == 1)

    def is_all_X_same(self,X:np.ndarray):
        """
        Judge if all data of X is class
        """
        return np.all(np.all(X == X[0,:], axis = 1))

    def generate_tree(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray,A:set,depth:int):
        """
        Generate a sub tree or node \n
        Params:
        X : the dataset or subset of X, 2D array
        Y : the label datas of X, 1D array
        A : the set of unused attributes of X, set
        """
        if self.is_all_Y_in_one_class(Y):
            cls = Y[0]
            return Node(
                is_leaf = True,
                children = {},
                result = cls,
                rate = 1.,
                depth = depth + 1
            )
        if self.is_all_X_same(X) or\
            len(A)==0 or\
            depth + 1 >= self.max_depth:
            cls,p = mode(Y,return_percents=True)
            return Node(
                is_leaf = True,
                children = {},
                result = cls[0],
                rate = p[0],
                depth = depth + 1
            )
        else:
            attr = self.calculate_max_attribute(X,Y,W,A)
            A_ = set(A)
            A_.remove(attr)
            A_sub = set(A_)
            classes = np.unique(X[:,attr])
            X_subs,Y_subs,W_subs = self.split_XYW_by_attr(X,Y,W,attr)
            children = {}
            for i,cls in enumerate(classes):
                X_sub = X_subs[i]
                Y_sub = Y_subs[i]
                W_sub = W_subs[i]
                child = self.generate_tree(X_sub,Y_sub,W_sub,A_sub,depth+1)
                children[cls] = child
            cls,p = mode(Y,return_percents=True)
            return Node(
                is_leaf = False,
                children = children,
                attribute = attr,
                result = cls[0],
                rate = p[0],
                depth = depth + 1
            )

    def is_last_root(self,node:Node):
        """
        Judge if this node is last root node
        """
        for child in node.children.values():
            if child.is_leaf == False:
                return False    
        return True

    def _pre_prune(self,node:Node):
        if node.is_leaf == False:
            for key,child in node.children.items():
                node.children[key] = self._pre_prune(child)
            if self.is_last_root(node):
                results = []
                for child in node.children.values(): 
                    results.append(child.result)
                results = np.array(results)
                classes = np.unique(results)
                if classes.shape[0] == 1:
                    node.is_leaf = True
                    node.attr = None
                    node.children = {}
        return node

    def pre_prune(self,node:Node):
        """
        Prune the useless nodes
        """
        return self._pre_prune(node)

    def calculate_prune_gain(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray,node:Node):
        node.prune_gain = Y.shape[0] * self.calculate_impurity(Y,W)
        if node.is_leaf == False:
            attr = node.attr
            X_subs,Y_subs,W_subs = self.split_XYW_by_attr(X,Y,W,attr)
            for i,child in enumerate(node.children.values()):
                X_sub = X_subs[i]
                Y_sub = Y_subs[i]
                W_sub = W_subs[i]
                self.calculate_prune_gain(X_sub,Y_sub,W_sub,child)

    def _prune(self,node:Node,alpha:float):
        if node.is_leaf == False:
            for key,child in node.children.items():
                node.children[key] = self._prune(child,alpha)
            if self.is_last_root(node):
                impurities = []
                for child in node.children.values():
                    impurities.append(child.prune_gain)
                impurities_sum = np.sum(impurities)
                n_children = len(node.children)
                #计算损失函数的差值
                delta = -impurities_sum + node.prune_gain - alpha * (n_children - 1)
                if delta <= 0 and node.is_leaf == False:
                    node.is_leaf = True
                    node.attr = None
                    node.children = {}
        return node

    def prune(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray=None):
        """
        Prune the tree with alpha \n
        Params:
        X : The pruning input samples
        Y : The target values (class labels) as integers or strings
        W : Sample weights. If None, then samples are equally weighted
        """
        if self.prunning:
            if W is None: W = np.ones(X.shape[0])
            self.calculate_prune_gain(X,Y,W,self.tree)
            pruned_tree = self._prune(self.tree,self.alpha)
            return self.pre_prune(pruned_tree)
        else:
            return self.tree

    def _fit(self,X:np.ndarray,Y:np.ndarray,W:np.ndarray=None,pdata=None):
        if W is None: W = np.ones(X.shape[0])
        A = set(range(X.shape[1]))
        tree = self.generate_tree(X,Y,W,A,depth=-1)
        self.tree = self.pre_prune(tree)
        if pdata is None:
            self.tree = self.prune(X,Y,W)
        else:
            if len(pdata) == 2: 
                X_p,Y_p = pdata
                W_p = None
            elif len(pdata) == 3:
                X_p,Y_p,W_p = pdata
            else:
                ValueError("The shape of pruning data must be tuple:(X,Y) or tuple(X,Y,W)")
            self.tree = self.prune(X_p,Y_p,W_p)

    def fit(self,X:np.ndarray,Y:np.ndarray,sample_weight=None,pruning_data:tuple=None):
        """
        Fit this model with datas \n
        Params:
        X : The training input samples
        Y : The target values (class labels) as integers or strings
        sample_weight : Sample weights. If None, then samples are equally weighted
        pruning_data : The data for pruning. If None, then the input data will be used for pruning (tuple:(X,Y) or (X,Y,W))
        """
        if self.check_Xdim(X):
            if self.check_XY(X,Y):
                W = sample_weight
                pdata = pruning_data
                self._fit(X,Y,W,pdata)
            else:
                raise ValueError("The size of X and Y is not same, please check your data")
        else:
            dims = np.ndim(X)
            raise ValueError("Expected 2D array, got {}D array instead".format(dims))
    
    def search(self,X:np.ndarray,node:Node):
        """
        Recursively search decision tree
        X : X, 1D array
        node : Node to search
        """
        if node.is_leaf == False:
            attr = node.attr
            cls = X[attr]
            children = node.children
            if cls in children.keys():
                child = children[cls]
            else:
                return node.result
            return self.search(X,child)
        else:
            return node.result

    def _predict(self,X:np.ndarray):
        """
        X : X, 2D array
        """
        pred = np.zeros(X.shape[0],dtype=np.int32)
        tree = self.tree
        for i,x in enumerate(X):
            res = self.search(x,tree)
            pred[i] = res
        if pred.shape[0] == 1: pred = pred[0]
        return pred

    def score(self,X:np.ndarray,Y:np.ndarray,sample_weight:np.ndarray=None):
        """
        Return the mean accuracy on the given test data and labels \n
        Params:
        X : Test samples
        Y : True labels for X
        sample_weight : Sample weights. If None, then samples are equally weighted
        """
        W = sample_weight
        if W is None: W = np.ones(X.shape[0])
        Y_pred = self.predict(X)
        Y_comp = (Y_pred==Y).astype(np.int8)
        Y_comp_weighted = np.sum(W*Y_comp)
        sum = np.sum(W)
        return Y_comp_weighted / sum

    def _d_search(self,node:Node):
        """
        Recursively search the depth decision tree
        """
        if node.is_leaf == False:
            for child in node.children.values():
                self._d_search(child)
        else:
            if node.depth > self._depth:
                self._depth = node.depth

    def search_depth(self):
        """
        Calculate the max-depth of this decision tree
        """
        self._depth = 0
        self._d_search(self.tree)
        return self._depth

    @property
    def depth(self):
        return self.search_depth()
