import os
os.system('ldconfig')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten
import tensorflow.keras.backend as K



def r2_metric(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))



width = 1500
height = 1200
channels = 7

model = Sequential([
    keras.Input(shape=(None,height,width,channels)),
    layers.ConvLSTM2D(filters=15, kernel_size=(3, 3), padding="same", return_sequences=True),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding="same", return_sequences=True),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(filters=60, kernel_size=(3, 3), padding="same", return_sequences=False),
    layers.BatchNormalization(),
    layers.Conv2D(filters=2,kernel_size=(3,3),activation="sigmoid",padding="same"),
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse',metrics='MeanSquaredError')
model.summary()

from skimage.measure import block_reduce 



X = []
Y = []
path_prefix = 'D:\\windpath\\windpath\\12_sample\\klam21_'
for i in range(1, 2): #9
    for h in range(1, 2): #4
        for s in range(0,2): #6
            for d in range(0, 360, 60): #360
                height = np.full((1200,1500),h)
                speed = np.full((1200,1500), s)
                direction = np.full((1200,1500),d)
                uz_zero = np.full((1200,1500),0)
                vz_zero = np.full((1200,1500),0)
                path = path_prefix + "0000" +"{:02d}".format(i) +"_R010_H0"+ str(h) +".0_S00"+str(s)+".00_D"+"{:003d}".format(d)
                landuse = np.loadtxt(path +'/landuse.txt', delimiter = ' ', skiprows = 6, dtype = 'int')
                terrain = np.loadtxt(path +'/terrain.txt', delimiter = ' ', skiprows = 6, dtype = 'int')
                #landuse = block_reduce(landuse, block_size=(20, 20), func=np.mean)
                #terrain = block_reduce(terrain, block_size=(20, 20), func=np.mean)
                stacked_layer_input = []
                stacked_layer_input.append(np.dstack((uz_zero,vz_zero,landuse,terrain,direction,speed,height)))
                for j in range(360, 3960, 360):
                    uz = np.loadtxt(path+'/result/'+"0000" +"{:02d}".format(i) +"_R010_H0"+ str(h) +".0_S00"+str(s)+".00_D"+"{:003d}".format(d)+'_uz00'+"{:04d}".format(j)+'.dw', skiprows = 8, dtype = 'int', encoding='latin-1')
                    vz = np.loadtxt(path+'/result/'+"0000" +"{:02d}".format(i) +"_R010_H0"+ str(h) +".0_S00"+str(s)+".00_D"+"{:003d}".format(d)+'_vz00'+"{:04d}".format(j)+'.dw', skiprows = 8, dtype = 'int', encoding='latin-1')
                    #uz = block_reduce(uz, block_size=(20, 20), func=np.mean)
                    #vz = block_reduce(vz, block_size=(20, 20), func=np.mean)
                    if j == 3600:
                        stacked_layer_output = np.dstack((uz,vz))
                    else:
                        stacked_layer_input.append(np.dstack((uz,vz,landuse,terrain,direction,speed,height)))
                X.append(stacked_layer_input)
                Y.append(stacked_layer_output)


X = np.array(X)
Y = np.array(Y)
X.shape, Y.shape

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
history = model.fit(x=X, y=Y, validation_split=0.2, epochs=10, callbacks=[early_stop])