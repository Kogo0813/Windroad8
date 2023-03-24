from PIL import Image
import numpy as np
import tensorflow as tf
tf.config.list_physical_devices('GPU')

filepath = 'D:\\windpath\\windpath\\12_sample/'

# load tiff file
im1 = Image.open(filepath + 'Building_000002.tif')
im2 = Image.open(filepath + 'Landuse_000002.tif')
im3 = Image.open(filepath + 'Terrain_000002.tif')


# resize size -> landuse 이미지의 크기가 다른 것과 다름
im2 = im2.resize((im1.width, im1.height))


# convert to numpy array
arr1 = np.array(im1)
arr2 = np.array(im2)
arr3 = np.array(im3)


# stacked arrays
stacked = np.stack([arr1, arr2, arr3], axis=0)
dstacked = np.dstack([arr1, arr2, arr3])

# create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(stacked)
dataset


# another data
