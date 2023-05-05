# load package
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import keras
from osgeo import gdal
import tensorflow as tf
from tensorflow.python.client import device_lib
# dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten
import tensorflow.keras.backend as K
from keras.utils import get_custom_objects

# set multi-gpu
tf.config.list_physical_devices('GPU')
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(),devices=["/gpu:0", "/gpu:1", "/gpu:2"]) # hierarchical로 변경

# metrics
def r2_metric(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# model_1
with mirrored_strategy.scope():

    get_custom_objects().update({'r2_metric': r2_metric})
    width = 75
    height = 60
    channels = 5
    model = Sequential([
        keras.Input(shape=(None,height,width,channels)),
        layers.ConvLSTM2D(filters = 16, kernel_size=(3, 3), padding="same", return_sequences=True),
        layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True),
        layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True),
        layers.ConvLSTM2D(filters=2, kernel_size=(3, 3), padding="same", return_sequences=False)
    ])
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse',metrics=[r2_metric])
    model.summary
    

# 풍향, 풍속에서 uz, vz추출하는 함수
def uz_vz_extract(d, s):
    import numpy as np
    from math import pi
    if d == 60:
        d = pi/3
    elif d == 120:
        d = 2*pi/3
    elif d == 180:
        d = pi
    elif d == 240:
        d = 4*pi/3
    else: d = 5*pi/3
    
    vz = np.sqrt(s)*np.sin(d)
    uz = np.sqrt(s)*np.cos(d)

    return uz, vz


# 일단은 기본 모델에서 7개로 3개를 예측하는 모델만 
from sklearn.preprocessing import MinMaxScaler

filepath = 'D:\\windpath\\data_from_s/'
X = []
Y = []
for i in range(2, 13):
  for h in range(1, 4):
    for s in range(0,6):
      for d in range(0, 360, 60):
        if i==3 and h==1 and s ==3 : 
          continue
        if i==3 and h==3 and s ==1 and d==300: 
          continue
        if i==3 and h==3 and s ==5 and d==120: 
          continue
        if i==8 and h==1 and s ==3 and d==120: 
          continue
        if i==9 and h==2 and s ==3 and d==300: 
          continue ## 이 파일들 이상해서 그냥 안쓸려고 합니다.

        
        height = np.full((60,75),h) # 높이는 1부터 3까지 1~3사이
        speed = np.full((60,75), s/5) # 속도는 정규화해서 표현 0~1사이
        direction = np.full((60,75),d) # 0~360 사이(0 = 2pi)
        scaler_3 = MinMaxScaler()
        area = scaler_3.fit_transform(height + speed + direction) # 3가지 파라미터 합친거


        uz_0, vz_0 = uz_vz_extract(d,s) # 초기 풍향, 풍속 조정
        uz_zero = np.full((60,75),uz_0)
        vz_zero = np.full((60,75),vz_0)

        scaler_terrain = MinMaxScaler()
        scaler_landuse = MinMaxScaler()

        landuse = gdal.Open(filepath + "input\\Landuse_"+"{:006d}".format(i)+".tif", gdal.GA_ReadOnly)
        landuse = landuse.GetRasterBand(1).ReadAsArray()
        terrain = gdal.Open(filepath + "input\\Terrain_"+"{:006d}".format(i)+".TIF", gdal.GA_ReadOnly)
        terrain = terrain.GetRasterBand(1).ReadAsArray()
        
        landuse = landuse[:60,:75]
        terrain = terrain[:60,:75]
        terrain = scaler_terrain.fit_transform(terrain)
        landuse = scaler_landuse.fit_transform(landuse)

        stacked_layer_input = []
        stacked_layer_input.append(np.dstack((uz_zero,vz_zero,landuse,terrain,area)))

        scaler_z = MinMaxScaler()

        for j in range(360, 3960, 360):
            uz = np.loadtxt('D:\\windpath\\data_from_s/' + "0000" +"{:02d}".format(i) +"_R010_H0"+ str(h) +".0_S00"+str(s)+".00_D"+"{:003d}".format(d)+'_uz00'+"{:04d}".format(j)+'.dw', skiprows = 8, dtype = 'int', encoding='latin-1')
            vz = np.loadtxt('D:\\windpath\\data_from_s/' + "0000" +"{:02d}".format(i) +"_R010_H0"+ str(h) +".0_S00"+str(s)+".00_D"+"{:003d}".format(d)+'_vz00'+"{:04d}".format(j)+'.dw', skiprows = 8, dtype = 'int', encoding='latin-1')
            uz = uz[:60,:75]
            vz = vz[:60,:75]
            uz = scaler_z.fit_transform(uz)
            vz = scaler_z.fit_transform(vz)
            if j == 2880: # 3시점을 예측하도록 변경
                stacked_layer_output = np.dstack((uz,vz))
            elif j >= 2880:
                stacked_layer_output.append(np.dstack((uz,vz))) # 예측하는 3시점의 결과 저장    
            else:
                stacked_layer_input.append(np.dstack((uz,vz,landuse,terrain,area)))
        X.append(stacked_layer_input)
        Y.append(stacked_layer_output)

X_1 = np.array(X)
Y_1 = np.array(Y)
X_1.shape, Y_1.shape

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_1, Y_1, test_size=0.2, shuffle=True)

with mirrored_strategy.scope():
    early_stop = EarlyStopping(monitor='r2_metric',min_delta=0.0001, patience=3, verbose=1, mode='min',restore_best_weights=True)
    history = model.fit(x=x_train, y=y_train,epochs=400, callbacks=[early_stop], validation_data=(x_val, y_val), batch_size = 2)