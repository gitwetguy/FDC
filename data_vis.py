import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

def read_data(path):
    df = pd.read_csv(path, sep=',',low_memory=False)
    
     # filling nan with mean in any columns
    
    
    return df
def find_coi(data,Y):
    
    data_list = list(range(data.shape[1]))
    res = set(data_list).difference(set(Y))
    return list(res)
    
def fill_nan_value(df):
    for j in range(df.shape[1]):  
        print(j)
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
        #df.iloc[:,4:] = df.iloc[:,4:]*100
        # another sanity check to make sure that there are not more any nan
    print(df.isnull().sum())
    print(df.head())
        
    return df

def show_corr(data,group,corr='spearman'):
    plt.matshow(data.iloc[:,group].corr(method=corr),vmax=1,vmin=-1,cmap='PRGn')
    plt.title(corr, size=15)
    plt.xticks(range(len(group)),labels=map(str,group))
    plt.yticks(range(len(group)),labels=map(str,group))
    plt.colorbar()
    plt.show()

#print num:column map
def df_col_map(data):
    num = 0
    for i in data.columns:
        print("{} : {}".format(num,i))
        num = num+1
        
def drop_and_showcorr(data,drop_list):
    data = data.drop(data.iloc[:,[5]],axis=1)
    show_corr(group=range(data.shape[1]))
    #print(data.describe())
    
#draw_trend
def draw_trend(df,groups,up=0,down=200):
    figsize(20,15) 
    i=1
    Values = df.values
    cols = groups
    for group in groups:
        plt.subplot(len(cols), 1, i)
        plt.plot(Values[up:down, group],linewidth=3)
        plt.title("{}-{}".format(group,df.columns[group]), y=0.80, loc='right')
        #plt.xticks(range(0,4020,20))
        i += 1
    plt.show()
    print('\n')
    
def draw_trend_anomaly(df,broken,groups,bound=200):
    figsize(20,15) 
    i=1
    Values = df.values
    cols = groups
    for group in groups:
        plt.subplot(len(cols), 1, i)
        plt.plot(Values[:bound, group],linewidth=3)
        plt.plot(broken[:bound, group], linestyle='none', marker='X', color='red', markersize=12)
        plt.title(df.columns[group], y=0.80, loc='right')
        #plt.xticks(range(0,4020,20))
        i += 1
    #plt.show()
    print('\n')
   