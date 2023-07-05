import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

pwd = os.path.dirname(os.path.realpath(__file__))

def save_pkl(data, pkl):
    with open(pkl, 'wb') as f:
        pickle.dump(data, f)
    return 

def load_pkl(pkl):
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    return data

def feature(featread):
    feat_dict = {}
    for key in featread.index:
        feat_dict[key] = featread.loc[key, :].values
    return feat_dict

def FP_Encode(data_train, feat_dict, On='On', Off='Off'):
    ont = data_train.loc[:, On].values
    offt = data_train.loc[:, Off].values
    train_target = np.array([[ont[i][j] + offt[i][j] for j in range(20)] for i in range(ont.shape[0])])
    X_train = np.vstack(
        [[np.vstack([[feat_dict[train_target[i][j]]] for j in range(20)])] for i in range(train_target.shape[0])])
    X_train = X_train.reshape(X_train.shape[0], -1)
    return X_train

def kfold_xgb_pred(model_path, X, y=None, kfold=5):
    kmodels = pickle.load(open(model_path, 'rb'))
    results = []
    for i in range(kfold):
        y_hat = kmodels['model_{}'.format(i)].predict_proba(X)[:, 1]
        results.append(y_hat)
    results = np.array(results)
    y_hat = results.mean(axis=0)
    return y_hat

def CRISOT_FP(model_path, dataread, feat_dict='default', On='On', Off='Off'):
    if feat_dict == 'default':
        featread = load_pkl(os.path.join(pwd, 'models/crisot_fingerprint_encoding.pkl'))
        feat_dict = feature(featread)
    X = FP_Encode(dataread, feat_dict, On, Off)
    y_hat = kfold_xgb_pred(model_path, X)
    return y_hat




