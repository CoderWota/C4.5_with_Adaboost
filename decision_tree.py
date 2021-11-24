import sys
sys.path.append(".")
import dctree.core
import dctree.processing
import numpy as np
import pandas as pd

class DecisionTreeClassifier:
    core_cls = dctree.core.DecisionTreeClassifier

    def __init__(self,criterion="gini",max_depth=np.inf,alpha:float=0.):
        """
        A C4.5 based decision tree classifier for pandas \n
        Params:
        criterion : The function to measure the quality of a split {"gini", "entropy"}
        max_depth : The maximum depth of the tree (pre-prunning)
        alpha : The alpha value for post-prunning, if alpha bigger than zero, then start prunning
        """
        self.core_model = DecisionTreeClassifier.core_cls(criterion,max_depth,alpha)
        self.tree = None
    
    def fit(self,X_df:pd.DataFrame,Y_df:pd.Series,cv_index,max_splits=2,pruning_data=None):
        """
        Fit this model with datas \n
        Params:
        X_df : The training input samples (pd.DataFrame);
        Y_df : The target values (class labels) as integers or strings (pd.Series)\n
        cv_index : The index of continuous variables (if all : "all")
        max_splits : Number of splits, split continuous variable max to 2^N\n
        pruning_data : The data for pruning. If None, then the input data will be used for pruning (tuple:(X,Y) or (X,Y,W))
        """
        X_df = X_df.copy()
        Y_df = Y_df.copy()
        X_df,Y_df,X_slices,X_reversed_slices,Y_slices,Y_reversed_slices,code_to_col,col_to_code =\
             dctree.processing.encoding(X_df,Y_df,cv_index,max_splits)
        X = X_df.values
        Y = Y_df.values
        self.core_model.fit(X,Y)
        self.cv_index = cv_index
        self.code_to_attr_X = X_slices
        self.attr_to_code_X = X_reversed_slices
        self.code_to_attr_Y = Y_slices
        self.attr_to_code_Y = Y_reversed_slices
        self.code_to_col_X = code_to_col
        self.col_to_code_X = col_to_code
        self.fitted = True
        tree = self.core_model.tree
        self.tree = dctree.processing.decoding_tree(
            tree,
            self.code_to_col_X,
            self.code_to_attr_X,
            self.code_to_attr_Y
        )
        self.depth = self.core_model.depth

    def predict(self,X_df:pd.DataFrame):
        """
        Predict class for X \n
        Params: 
        X_df : The input samples (pd.DataFrame)
        """
        cv_index = self.cv_index
        X_slice = self.attr_to_code_X
        Y_slice = self.code_to_attr_Y
        X_coded = dctree.processing.encoding_input(X_df,cv_index,X_slice)
        X = X_coded.values
        raw_output = self.core_model.predict(X)
        out = dctree.processing.decoding_predicts(raw_output,Y_slice)
        return out

    def score(self,X_df:pd.DataFrame,Y_df:pd.Series):
        """
        Return the mean accuracy on the given test data and labels \n
        Params:
        X_df : Test samples (pd.DataFrame)
        Y_df : True labels for X (pd.Series)
        """
        cv_index = self.cv_index
        X_slice = self.attr_to_code_X
        Y_slice = self.attr_to_code_Y
        X_coded = dctree.processing.encoding_input(X_df,cv_index,X_slice)
        Y_coded = dctree.processing.encoding_label(Y_df,Y_slice)
        X = X_coded.values
        Y = Y_coded
        out = self.core_model.score(X,Y)
        return out

    def print(self):
        """
        Prints the tree model to a stream
        """
        tree = self.tree
        res = []
        dot = "|---"
        side = "|   "
        def _print(node:dctree.core.Node):
            if node.is_leaf == False:
                frame = node.depth * side + dot + node.attr + " : "
                for key,val in node.children.items():
                    string = frame + key + "\n"
                    res.append(string)
                    _print(val)
            else:
                frame = node.depth * side + dot + " : "
                string = frame + node.result + " ratio: %.2f"%node.rate + "\n"
                res.append(string)
        _print(tree)
        res_str = "".join(res)
        print(res_str)
        print("Depth: {}".format(self.depth))

    def plot(self):
        pass

    def save(self,filename):
        """
        Save this model to a file
        Param: \n
        filename : (name).pkl
        """
        import pickle
        with open(filename,"wb") as f:
            pickle.dump(self,f)

if __name__ == '__main__':
    df = pd.read_csv("/Users/shakalaka/Documents/Data/Credit.csv",index_col=0)
    df = df.sample(frac=1)
    x_df = df.drop("Student",axis=1)
    y_df = df["Student"]
    x_train = x_df[:200]
    y_train = y_df[:200]
    x_test = x_df[200:]
    y_test = y_df[200:]
    cv_index = ["Income","Limit","Rating","Cards","Age","Education","Balance"]
    model = DecisionTreeClassifier(criterion="entropy",alpha=4)
    model.fit(x_train,y_train,cv_index,max_splits=2)
    model.print()
    print(model.score(x_test,y_test))