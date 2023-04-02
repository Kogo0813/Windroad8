from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten, Conv2D, MaxPooling3D, Dropout, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
import tensorflow as tf
import keras
import glob
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

# tif file load
filepath = 'D:\\windpath\\windpath\\12_sample\\'

ex_tif = gdal.Open(filepath + 'Building_000001.tif',
                   gdal.GA_ReadOnly)  # tif file load
print(ex_tif.GetMetadata())

# 밴드 데이터 보기
band = ex_tif.GetRasterBand(1)
values = band.ReadAsArray()
values.shape

# example image tiff data layer
building = gdal.Open(filepath + 'Building_000001.tif', gdal.GA_ReadOnly)
landuse = gdal.Open(filepath + 'Landuse_000001.tif', gdal.GA_ReadOnly)
terrain = gdal.Open(filepath + 'Terrain_000001.tif', gdal.GA_ReadOnly)


def layer_tiff(im1, im2, im3):
    x_size = im1.RasterXSize
    y_size = im1.RasterYSize

    projection = building.GetProjection()

    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create('output.tif', x_size, y_size, 3, gdal.GDT_Float32)
    output.SetProjection(projection)

    band1 = im1.GetRasterBand(1)
    output.GetRasterBand(1).WriteArray(band1.ReadAsArray())

    band2 = im2.GetRasterBand(1)
    output.GetRasterBand(2).WriteArray(band1.ReadAsArray())

    band3 = im3.GetRasterBand(1)
    output.GetRasterBand(3).WriteArray(band1.ReadAsArray())

    return output.ReadAsArray()


image_layer = layer_tiff(building, landuse, terrain)
image_layer


image_layer = np.resize(image_layer, (3, 1200, 1500))  # 이런식으로 줄여도 되는지
image_layer.shape  # 샘플마다 이미지 레이어는 다르지만 샘플안에서는 전부 동일함

# 지역풍 풍향, 풍속 데이터
wind_1 = np.full((1200, 1500), 0)
wind_2 = np.full((1200, 1500), 0)
height = np.full((1200, 1500), 1)

int_layer = np.concatenate((wind_1.reshape(1, 1200, 1500), wind_2.reshape(1, 1200, 1500),
                            height.reshape(1, 1200, 1500)), axis=0)

# image + int
total_layer = np.concatenate((image_layer, int_layer), axis=0)

total_layer.shape

# label data
uz_filepath = filepath + 'klam21_000001_R010_H01.0_S000.00_D000/result/'

uz_filelist = glob.glob(uz_filepath + '*uz*')
vz_filelist = glob.glob(uz_filepath + '*vz*')

print(" uz파일리스트의 길이: ", len(uz_filelist),
      '\n', "vz파일리스트의 길이: ", len(vz_filelist))

# label data load
np.loadtxt(uz_filelist[0], skiprows=8, dtype='int', encoding='latin-1').shape
uz_list = [f'uz_{i}' for i in range(len(uz_filelist))]
vz_list = [f'vz_{i}' for i in range(len(vz_filelist))]

for i, filename in enumerate(uz_filelist):  # load uz data
    uz_list[i] = np.loadtxt(filename, skiprows=8,
                            dtype='int', encoding='latin-1')

for i, filename in enumerate(vz_filelist):  # load vz data
    vz_list[i] = np.loadtxt(filename, skiprows=8,
                            dtype='int', encoding='latin-1')

# uz layer 쌓기
uz_layer = np.concatenate(
    (uz_list[0].reshape(1, 1200, 1500), uz_list[1].reshape(1, 1200, 1500)), axis=0)
for i in range(2, len(uz_filelist)):
    uz_layer = np.concatenate(
        (uz_layer, uz_list[i].reshape(1, 1200, 1500)), axis=0)
print(uz_layer.shape)

# vz layer 쌓기
vz_layer = np.concatenate(
    (vz_list[0].reshape(1, 1200, 1500), vz_list[1].reshape(1, 1200, 1500)), axis=0)
for i in range(2, len(vz_filelist)):
    vz_layer = np.concatenate(
        (vz_layer, vz_list[i].reshape(1, 1200, 1500)), axis=0)
print(vz_layer.shape)


time_steps = 10
width = 1500
height = 1200
channels = 6

model = Sequential([
    keras.Input(shape=(None, 1, height, width)),
    layers.ConvLSTM2D(filters=32, kernel_size=(
        3, 3), padding="same", return_sequences=True),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(filters=64, kernel_size=(
        3, 3), padding="same", return_sequences=True),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(filters=128, kernel_size=(
        3, 3), padding="same", return_sequences=True),
    layers.BatchNormalization()
])

# 모델 컴파일
model.compile(optimizer='adadelta', loss='mse', metrics=['r2'])
model.summary()

model.fit(total_layer, uz_layer[:6, :, :], epochs=10, batch_size=1, verbose=1)
