#!/usr/bin/env python
# coding: utf-8

#import ipdb
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


def cleanup_text(x):
    #converting all to lowercase
    x = x.lower()
    # replace commas, normalize spacing
    x = " ".join(x.replace(","," ").split(" "))
    return x

#one hot encoding features 
def one_hot_encode_text(data,text_label,min_frac=0.01):
    df=data.copy()
    df[text_label]=df[text_label].fillna("___")#replacing NA
    # cleanup
    df[text_label]=df[text_label].apply(cleanup_text)
    #obtaining a list of all strings for variable
    allsentences=df[text_label]
    #instantiating function to creat dummy variable per word that appears a minimum number of times
    vectorizer=CountVectorizer(min_df=min_frac)
    #note this works for each word of each string in the list 
    X=vectorizer.fit_transform(allsentences)
    #generating dummy variable count matrix ,columns are words, rows obs
    df[vectorizer.get_feature_names_out()]=X.toarray() #adding dummyvariables to df
    # drop original text feature
    df = df.drop(columns=text_label)
    return df




def label_encode_text(data,text_label,cum_sum=0.80):
    """converting string categorical features into numerical categorical features"""
    df=data.copy()
#     df[text_label]=df[text_label].fillna("___")#replacing NA
    # cleanup
    df[text_label]=df[text_label].fillna("___").apply(cleanup_text)
    #identifying the most infrequent ones to rename as such
    words_infrequent=list(df[text_label].value_counts()[((df[text_label].value_counts()/df.shape[0]).cumsum()>cum_sum)].index)
    df.loc[df[text_label].isin(words_infrequent),text_label]="infrequent"
    enc=LabelEncoder()
    df[text_label]=enc.fit_transform(df[text_label])
    code=list(list(vars(enc).items())[0][1])#storing labels 
    return code, df




#reading file wrapper function with options for one hot, label encoding for text strings, and label encoding
#for categorical variables 
def read_file(filename,label,text_features,one_hot_encode,ohc_min_frac=0.01,le_cum_sum=0.80):
    """read file into pandas data frame, optionally onehot encode text features and label encode categorical data.

    text_features: list of text features
    one_hot_encode: list of booleans same length of text_features to determine if you will onehotencode

    return X,y
    """
    # reading in data
    input_data=pd.read_csv(filename) 
    
    # remove missing,other,unable to obtain
    input_data = input_data.loc[~input_data['ethnicity'].isin([
        "UNKNOWN","OTHER","UNABLE TO OBTAIN"
    ])]
    encodings={}
    for (v,c) in zip(text_features,one_hot_encode):  
        if(c):
            print('One Hot Encoding', v)
            input_data=one_hot_encode_text(input_data,v,min_frac=ohc_min_frac)
        else:
            print('Label Encoding',v)
            codes,input_data=label_encode_text(input_data,v,cum_sum=le_cum_sum)
            encodings[v]=codes
    #converting the categorical data into numerical caetgorical data instead of strings
    X=input_data.drop(columns=label,axis=1)
    for c in X.select_dtypes(['object','category']).columns:
        print(f'Label Encoding {c},not pre-specified')
        le=LabelEncoder() #
        X[c]=le.fit_transform(X[c].fillna("___"))#replacing missing with that first
        encodings[c] = list(list(vars(le).items())[0][1])
        
    y = input_data[label].astype(int)
    
    return X, y, encodings