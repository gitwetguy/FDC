import numpy as np
import matplotlib.pyplot as plt


# 0 : Unnamed: 0
# 1 : Latch:_LA_269353054_DevicePositionXOnLFAtP1;_CEID15651.4
# 2 : Latch:_LA_269353055_DevicePositionYOnLFAtP1;_CEID15651.5
# 3 : Latch:_LA_269353094_DispenseProcessAvePress;_CEID15651.7
# 4 : Latch:_LA_269353095_DispenseProcessAveZHeight;_CEID15651.8
# 5 : Latch:_LA_251920580_DispenserPressure_ECRO;_CEID15651.9
# 6 : Latch:_LA_251920465_DispenseHeightToStrip_ECRO;_CEID15651.10
# 7 : Latch:_LA_269353066_DiePositionOnStripX;_CEID15652.4
# 8 : Latch:_LA_269353067_DiePositionOnStripY;_CEID15652.5
# 9 : Latch:_LA_252904019_WaferID;_CEID15653.4
# 10 : Latch:_LA_269353065_PostbondEpoxyCoverageCheckData;_CEID15652.10
# 11 : Sig:_BondProcess_AveBondForce_(post_Step)
# 12 : Sig:_BondProcess_AveBondZHeight_(post_Step)
# 13 : Latch:_LA_269353101_PickProcessAvePickForce;_CEID15653.7
# 14 : Latch:_LA_269353102_PickProcessAveNeedleTopHeight;_CEID15653.8
# 15 : Sig:_BondProcess_PBIEpoxyCoverage_(post_Step)
# 16 : Latch:_LA_269353097_BondProcessAveBondForce;_CEID15652.6
# 17 : Latch:_LA_269353098_BondProcessAveBondZHeight;_CEID15652.7
# 18 : Latch:_LA_252314679_BondForceNotHighForceArray_ECRO;_CEID15652.8
# 19 : Latch:_LA_252314736_BondDistanceToBondPosition_ECRO;_CEID15652.9
# 20 : Latch:_LA_269353069_DiePlacementOnStripX;_CEID15652.11
# 21 : Latch:_LA_269353070_DiePlacementOnStripY;_CEID15652.12
# 22 : Latch:_LA_269353071_DiePlacementOnStripTheta;_CEID15652.13
# 23 : Latch:_LA_252313626_PickForceD180_ECRO;_CEID15653.9
# 24 : Latch:_LA_252313630_PickNeedleTopHeight_ECRO;_CEID15653.10
# 25 : Latch:_LA_252904019_WaferID;_CEID15653.4.1
# 26 : Sig:_BondProcess_PBIDiePlacementOnStripTheta_(post_Step)
# 27 : Sig:_BondProcess_PBIDiePlacementOnStripX_uM_(post_Step)
# 28 : Sig:_BondProcess_PBIDiePlacementOnStripY_uM_(post_Step)

def gen_dataset(data,y_column_list,x_window_size,y_window_size,X_col,y_col):
    X_train = []   #預測點的前 60 天的資料
    y_train = []   #預測點
    for i in range(x_window_size, data.shape[0]-y_window_size,y_window_size):
        X_train.append(data[i-x_window_size:i, :])
        y_train.append(data[i:i+y_window_size, y_col ])

    X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
    
    
    
    X_train = np.reshape(X_train, (X_train.shape[0], x_window_size,data.shape[1] ))
    print(y_train.shape)
    if y_window_size == 1:
        y_train = np.reshape(y_train,(y_train.shape[0],y_train.shape[2]))
    else:
        y_train = np.reshape(y_train,(y_train.shape[0],y_window_size,y_train.shape[2]))
    print("Gen data info:")
    print("X_data_shape:{}".format(X_train.shape))
    print("y_data_shape:{}".format(y_train.shape))
    print("\n")
          
    
    return X_train,y_train

def data_scaling(data,X_win_size,y_win_size,mode,X_col,y_col):
    
    assert mode == 'robust' or mode == 'minmax','Wrong mode name'
    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler,RobustScaler
    #data:df type
    
    #80o/o data
    train_count = int(data.shape[0]/24*0.8)*24
    print(train_count*24)
    
    training_set=data.iloc[:train_count,:].values
    testing_set=data.iloc[train_count:,:].values
    
    print(training_set.shape)
    print(testing_set.shape)
    
    
    print("Mode:{}\n".format(mode))
    
    if mode =='robust':
        
         #RobustScaler
        
        sc_train = RobustScaler()
        X_train_scaled = sc_train.fit_transform(training_set)    

        sc_test = RobustScaler()
        X_test_scaled = sc_test.fit_transform(testing_set)
    
    elif mode == 'minmax':
        
        #MinMaxScaler
        
        #sc_train = MinMaxScaler(feature_range=(0,1))
        sc_train = MinMaxScaler(feature_range=(0,1))
        X_train_scaled = sc_train.fit_transform(training_set)    

        sc_test = MinMaxScaler(feature_range=(0,1))
        X_test_scaled = sc_test.fit_transform(testing_set)
        
        
    
    X_column=X_col
    y_column=y_col
    
    #gen dataset
    train_X,train_y = gen_dataset(X_train_scaled,y_column,X_win_size,y_win_size,X_column,y_column)
    test_X,test_y = gen_dataset(X_test_scaled,y_column,X_win_size,y_win_size,X_column,y_column)
    org_test_x,org_test_y= gen_dataset(testing_set,y_column,X_win_size,y_win_size,X_column,y_column)
    
    print("Data shape info:")
    print("origin_test_data:{}".format(testing_set.shape))
    print("X_train:{}".format(train_X.shape))
    print("y_train:{}".format(train_y.shape))
    print("X_test:{}".format(test_X[0]))
    print("y_test:{}".format(test_y[0]))
    print("O_X_test:{}".format(org_test_x.shape))
    print("O_y_test:{}".format(org_test_y.shape))
    
    #return original test data, 
    
    return training_set,testing_set,train_X,train_y,test_X,test_y,sc_test,org_test_y