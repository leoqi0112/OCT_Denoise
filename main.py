import warnings
warnings.filterwarnings('ignore')

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from denoiseg.models import DenoiSeg, DenoiSegConfig
from denoiseg.utils.misc_utils import combine_train_test_data, shuffle_train_data, augment_data
from denoiseg.utils.seg_utils import *
from denoiseg.utils.compute_precision_threshold import measure_precision

from csbdeep.utils import plot_history

import urllib
import os
import zipfile
import cv2
import json
from pathlib import Path
import skimage.segmentation as seg
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,LabelBinarizer
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

model_name = 'Insert_Model_Name'
model = DenoiSeg(None, model_name)

sl = 21
test_noisy_images = np.load('images_dataset.npy')
input_images = test_noisy_images[sl]
prediction = model.predict(input_images,'YX')

prediction_exp = np.exp(prediction[..., 1:])
prediction_softmax = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
prediction_img = np.argmax(prediction_softmax,2)


plt.figure(figsize=(12,12))
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.suptitle('Result visualizations')
plt.subplot(2,2,1)
plt.title('Input image',fontsize=12)
plt.imshow(input_images,cmap='gray')
plt.subplot(2,2,2)
plt.title('Denoised image',fontsize=12)
plt.imshow(prediction[:,:,0],cmap='gray')
plt.subplot(2,2,3)
plt.title('Prediction',fontsize=12)
plt.imshow(prediction_img)
plt.subplot(2,2,4)
plt.title('GT',fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.95)