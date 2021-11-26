from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.vae import VAE
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
# Import the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,GRU, Input, ConvLSTM2D, Bidirectional,BatchNormalization
from tensorflow.keras import Input,layers
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,Nadam,Adamax

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
import math
import json,os
from IPython.core.pylabtools import figsize
FDC_path = r"E:\FDC\dataset"

figsize(10,10) 
import data_vis as dv
import importlib
importlib.reload(dv)


FDC_2021Data = dv.read_data(os.path.join(FDC_path,'data_all_exclude_miss.csv'))
Y_LIST = [1,2,3,4,5,19,20,22,23,26,27,28,35,36,37,44]

X_LIST = dv.find_coi(FDC_2021Data,Y_LIST)
FDC_2021Data = FDC_2021Data.drop(FDC_2021Data.iloc[:,X_LIST],axis=1)
dv.df_col_map(FDC_2021Data)



def gen_dataset(data,win_size):
    raw_dataset = []   #預測點的前 60 天的資料
    #y_train = []   #預測點
    for i in range(win_size, data.shape[0],win_size):
        raw_dataset.append(data[i-win_size:i, :])
        
    raw_dataset = np.array(raw_dataset)
    print("raw_dataset_shape:{}".format(raw_dataset.shape))
    return raw_dataset

data = np.array(FDC_2021Data.iloc[:,5:15])
targets = np.array(FDC_2021Data.iloc[:,0])
from sklearn.preprocessing import MinMaxScaler,RobustScaler

scaler = MinMaxScaler((0,1))
X_scaled = scaler.fit_transform(data)

train_data = X_scaled


train_cnt = int((train_data.shape[0]*0.8))

X_train = train_data[:train_cnt,:]
X_test  = train_data[train_cnt:,:]
y_train = np.array(FDC_2021Data.iloc[:train_cnt,0])
y_test = np.array(FDC_2021Data.iloc[train_cnt:,0])

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 20000  # number of training points
    n_test = 2000  # number of testing points
    n_features = 300  # number of features

    

    # train VAE detector (Beta-VAE)
    clf_name = 'VAE'
    clf = VAE(encoder_neurons=[24,12,6], decoder_neurons=[6,12,24],
                 latent_dim=2, hidden_activation='relu',
                 output_activation='sigmoid', optimizer='adam',
                 epochs=1, batch_size=2048, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=False,
                 verbose=1, random_state=None, contamination=0.1,
                 gamma=1.0, capacity=0.0)
    clf.fit(X_train,y_train)
    #from joblib import dump, load

    # save the model
    #dump(clf, 'VAE.joblib')
    # load the model
    #clf = load('VAE.joblib')
    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print(clf.layers[0])