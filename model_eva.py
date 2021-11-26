import tensorflow as tf
import shutil,os
import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd# load and evaluate a saved model
from numpy import loadtxt
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,GRU, Input, ConvLSTM2D, Bidirectional,BatchNormalization
from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,Nadam


#For Training
def clean_folder(path):
        
    shutil.rmtree(path) 
    os.mkdir(path) 

def tensorboard_setting(name):

    logdir = os.path.join("./logs", name+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    
    return tensorboard_callback

def model_chk(path):
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=path,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)
    
    print("checkpoint save in {}".format(path))
    
    return checkpoint
    
def setup_tfboard(clean=False,log_name_type="exp"):
    #import tensorflow as tf
    if clean == False:
        print("Continue using current logs")
        callback = tensorboard_setting(log_name_type)
    else:
        print("Clean current logs") 
        clean_folder("logs")
        callback = tensorboard_setting(log_name_type)
    
    return callback

#For Testing
def prediction(sc,test,origin_data):
    
    #pred_list = []
    pred = model.predict(test)
    print(origin_data.shape)
    print(origin_data.shape)
    plt.plot(origin_data)
    plt.show()
    res = sc.inverse_transform(np.concatenate([origin_data,pred],axis=1))
    #pred_list.append(res[:,4:])
    return res

def cal_score(y_real,y_hat):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import math

    MAEScore = mean_absolute_error(y_real,y_hat)
    RMSEScore = math.sqrt(mean_squared_error(y_real,y_hat))

    return round(MAEScore,3),round(RMSEScore,3)

def draw_compare(real_data,pred_data):

    print(cal_score(real_data,pred_data))
    
    plt.plot(real_data,color = 'red',marker='.', label = 'Real MEM Used',linewidth=3)
    plt.plot(pred_data,color = 'blue', label = 'Predicted MEM Used',linewidth=3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.legend(fontsize='xx-large')
    plt.xlabel('Sequence',fontsize=20)
    plt.ylabel('24H Detection Trend Values',fontsize=20)
    plt.title('predicted values and actual values on test data',fontsize=20)
    plt.show()