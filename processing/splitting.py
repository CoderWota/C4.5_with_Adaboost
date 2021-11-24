import sys
sys.path.append(".")
from dctree.processing.cv_processor import CVProcessor
import pandas as pd
import numpy as np

def check_split_index(df,split_index):
    """
    Check the split index.
    If the split index's datatype is integer, then change it into string
    """
    index = np.asarray(split_index)
    if index.dtype == int:
        index = df.columns[index].values
    return index

def split_dataframe(df:pd.DataFrame,split_index:np.ndarray):
    """
    Split out the continuous variables from a dataframe \n
    Params:
    df : Pandas dataframe
    split_index : Indices of continuous variables
    """
    return df.loc[:,split_index].values

def coding_series(series:pd.Series):
    cat = series.astype("category").cat
    series_coded = cat.codes
    code_dict = dict(enumerate(cat.categories))
    reversed_dict = dict(zip(cat.categories,range(len(cat.categories))))
    return series_coded,code_dict,reversed_dict

def cut_slices(slices:np.ndarray):
    d = {}
    reversed_d = {}
    slices = np.insert(slices,0,-np.inf)
    slices = np.append(slices,np.inf)
    for i in range(len(slices)-1):
        array = np.asarray([slices[i],slices[i+1]])
        d[i] = array
        reversed_d[str(array)] = i
    return d,reversed_d

def trans_cv_to_dv(X:np.ndarray,Y:np.ndarray,n_splits=2):
    """
    Transform continuous variables to discrete variables
    """
    slices = []
    reversed_slices = []
    for i,X_column in enumerate(X.T):
        processor = CVProcessor(n_splits)
        processor.fit(X_column,Y)
        X_new_column = processor.predict(X_column)
        X[:,i] = X_new_column
        slice = processor.slices
        slice_dct,reversed_dct = cut_slices(slice)
        slices.append(slice_dct)
        reversed_slices.append(reversed_dct)
    return X.astype(int),slices,reversed_slices

def to_dict(index,slices):
    return dict(zip(index,slices))

def df_to_discrete(X_df:pd.DataFrame,Y_df:pd.Series,split_index,n_splits=2):
    """
    Transform the continuous variables into discrete variables with split index. \n
    Params:
    X_df : The input data with datatype of pandas.DataFrame
    Y_df : The label data with datatype of pandas.DataFrame
    split_index : The index of continuous variables (if all : "all")
    n_splits: Number of splits, split continuous variable max to 2^N
    Return:
    X_df : The coded input data
    Y_df : The coded label data
    X_slices : The code dictionary for output data (code->attribute)
    X_reversed_slices : The code dictionary for output data (attribute->code)
    Y_slices : The code dictionary for label data (code->attribute)
    Y_reversed_slices : The code dictionary for label data (attribute->code)
    """
    if split_index == "all": split_index = np.arange(X_df.shape[1])
    index = check_split_index(X_df,split_index)
    X_c = split_dataframe(X_df,index)
    Y_df,Y_slices,Y_reversed_slices = coding_series(Y_df)
    Y = Y_df.values
    X_c,X_slices,X_reversed_slices = trans_cv_to_dv(X_c,Y,n_splits)
    X_df[index] = X_c
    X_slices = to_dict(index,X_slices)
    X_reversed_slices = to_dict(index,X_reversed_slices)
    return X_df,Y_df,X_slices,X_reversed_slices,Y_slices,Y_reversed_slices
