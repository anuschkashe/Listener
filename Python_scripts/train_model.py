# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:13:33 2021

@author: User
"""
import tensorflow as tf
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
from sklearn.model_selection import train_test_split
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import glob
import random

from IPython.display import Audio, Image
from scipy.io import wavfile

def show_confusion_matrix(confusion, test_labels):
  """Compute confusion matrix and normalize."""
  confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
  axis_labels = test_labels
  ax = sns.heatmap(
      confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
      cmap='Blues', annot=True, fmt='.2f', square=True)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")

print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Maker Version: {mm.__version__}")

# @title [Run this] Util functions and data structures.

data_dir = 'C:/Users/User/Desktop/letitrock/eth_medical/Code_python/dataset/1_training'

#%%
test_files = os.path.abspath(os.path.join(data_dir, 'test/*/*.wav'))


print('functions and data structures created')

spec = audio_classifier.YamNetSpec(
    keep_yamnet_and_custom_heads=False,
    frame_step= audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
    frame_length= audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

#%%
train_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(data_dir, 'train'), cache=True)
train_data, validation_data = train_data.split(0.7)
test_data = validation_data
#%%
#test_data = audio_classifier.DataLoader.from_folder(
#    spec, os.path.join(data_dir, 'test'), cache=True)
batch_size = 16
#128
epochs = 8

print('Training the model')
model = audio_classifier.create(
    train_data,
    spec,
    validation_data,
    batch_size=batch_size,
    epochs=epochs, train_whole_model=True)
print('Evaluating the model')

confusion_matrix = model.confusion_matrix(train_data)
show_confusion_matrix(confusion_matrix.numpy(), train_data.index_to_label)
#%%
model.evaluate(test_data)

confusion_matrix = model.confusion_matrix(test_data)
show_confusion_matrix(confusion_matrix.numpy(), test_data.index_to_label)
#%%
# if you want to try another file just uncoment the line below
#random_audio = get_random_audio_file()
#show_bird_data(random_audio)

models_path = 'C:/Users/User/Desktop/letitrock/eth_medical/Code_python'
print(f'Exporing the TFLite model to {models_path}')

model.export(models_path, tflite_filename='my_birds_model.tflite')
