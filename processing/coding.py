import sys
sys.path.append(".")
from dctree.core import Node
import dctree.processing.splitting as splitting
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

def split_discrete_variables(df:pd.DataFrame,slices:dict):
    cv_index = np.array(list(slices.keys()))
    df = df.drop(cv_index,axis=1)
    return df

def coding_series(series:pd.Series):
    cat = series.astype("category").cat
    series_coded = cat.codes
    code_dict = dict(enumerate(cat.categories))
    reversed_dict = dict(zip(cat.categories,range(len(cat.categories))))
    return series_coded,code_dict,reversed_dict

def coding_dataframe(df:pd.DataFrame,slices:dict,rv_slices:dict):
    df_sub = split_discrete_variables(df,slices)
    index = df_sub.columns
    slices_sub = {}
    rv_sub = {}
    for col in df_sub.columns:
        s = df_sub[col]
        s_coded,s_dict,rv_dict = coding_series(s)
        df_sub[col] = s_coded
        slices_sub[col] = s_dict
        rv_sub[col] = rv_dict
    df[index] = df_sub
    slices.update(slices_sub)
    rv_slices.update(rv_sub)
    return df,slices,rv_slices

def coding_columns(index):
    code_to_col = dict(zip(np.arange(len(index)),index))
    col_to_code = dict(zip(index,np.arange(len(index))))
    return code_to_col,col_to_code

def convert_str_to_array(string:str):
    return np.fromstring(string[1:-1],sep=" ")

def coding_input_cv(X_df:pd.DataFrame,cv_index,X_reversed_slices:dict):
    if cv_index == "all": cv_index = np.arange(X_df.shape[1])
    cv_index = splitting.check_split_index(X_df,cv_index)
    slice = X_reversed_slices
    for col in cv_index:
        col_data = X_df[col].values
        for string_dict,value in slice[col].items():
            array_dict = convert_str_to_array(string_dict)
            idx = (np.less.outer(col_data,array_dict[1:]) &\
                 np.greater_equal.outer(col_data,array_dict[:-1])).squeeze()
            col_data[idx] = value
        X_df.loc[:,col] = col_data
    return X_df

def mapping_code_to_attr(X:np.ndarray,code_dict:dict):
    #"https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python"
    k = np.array(list(code_dict.keys()))
    v = np.array(list(code_dict.values()))

    sidx = k.argsort() #k,v from approach #1

    k = k[sidx]
    v = v[sidx]

    idx = np.searchsorted(k,X.ravel()).reshape(X.shape)
    idx[idx==len(k)] = 0
    mask = k[idx] == X
    out = np.where(mask, v[idx], 0)
    return out

def coding_input_dv(X_df:pd.DataFrame,cv_index,X_reversed_slices:dict):
    X_sub = X_df.drop(cv_index,axis=1)
    for col in X_sub.columns:
        mapping = X_reversed_slices[col]
        values = X_sub[col].values
        X_coded = mapping_code_to_attr(values,mapping)
        X_df.loc[:,col] = X_coded
    return X_df

def encoding_input(X_df:pd.DataFrame,cv_index,X_reversed_slices:dict):
    """
    Encoding the input data for predict.
    X_df : The input data with datatype of pandas.DataFrame
    cv_index : The index of continuous variables (if all : "all")
    X_reversed_slices : The dictionary of attributes to codes
    """
    X_cv_coded = coding_input_cv(X_df,cv_index,X_reversed_slices)
    return coding_input_dv(X_cv_coded,cv_index,X_reversed_slices).astype(int)

def encoding_label(Y_df:pd.Series,Y_reversed_slices:dict):
    """
    Encoding the input data for predict.
    X_df : The label data with datatype of pandas.DataFrame
    Y_reversed_slices : The dictionary of attributes to codes
    """
    values = Y_df.values
    return mapping_code_to_attr(values,Y_reversed_slices)

def decoding_predicts(Y_raw:np.ndarray,Y_slices:dict):
    """
    Decoding the label data for predict.
    """
    attrs = mapping_code_to_attr(Y_raw,Y_slices)
    return pd.Series(attrs)

def encoding(X_df:pd.DataFrame,Y_df:pd.Series,cv_index,n_splits=2):
    """
    Encoding the dataframe into C4.5 calculatable datas with continuous variables index. \n
    Params: \n
    X_df : The input data with datatype of pandas.DataFrame
    Y_df : The label data with datatype of pandas.DataFrame
    cv_index : The index of continuous variables (if all : "all")
    n_splits: Number of splits, split continuous variable max to 2^N \n
    Return:
    X_df : The coded input data
    Y_df : The coded label data
    X_slices : The code dictionary for output data (code->attribute)
    X_reversed_slices : The code dictionary for output data (attribute->code)
    Y_slices : The code dictionary for label data (code->attribute)
    Y_reversed_slices : The code dictionary for label data (attribute->code)
    code_to_col: The code dictionary for columns of output data (code->attribute)
    col_to_code : The code dictionary for columns of output data (attribute->code)
    """
    code_to_col,col_to_code = coding_columns(X_df.columns)
    X_df,Y_df,X_slices,X_reversed_slices,Y_slice,Y_reversed_slices =\
         splitting.df_to_discrete(X_df,Y_df,cv_index,n_splits)
    X_df,X_slices,X_reversed_slices = coding_dataframe(X_df,X_slices,X_reversed_slices)
    return X_df,Y_df,X_slices,X_reversed_slices,Y_slice,Y_reversed_slices,code_to_col,col_to_code

def decoding_tree(node:Node,code_to_col:dict,X_slices:dict,Y_slices:dict):
    if node.is_leaf == False:
        attr_name = code_to_col[node.attr]
        attr_res = Y_slices[node.result]
        new_children = {}
        for code,child in node.children.items():
            cls_name = X_slices[attr_name][code]
            if type(cls_name) == np.ndarray: cls_name = str(cls_name)
            new_children[cls_name] = decoding_tree(child,code_to_col,X_slices,Y_slices)
        return Node(
            is_leaf=False,
            children=new_children,
            attribute=attr_name,
            result=attr_res,
            rate=node.rate,
            depth=node.depth
        )
    else:
        attr_res = Y_slices[node.result]
        return Node(
            is_leaf=True,
            children={},
            result=attr_res,
            rate=node.rate,
            depth=node.depth
        )

