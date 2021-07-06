# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 21:47:34 2021

@author: User
"""
# Importing all required modules and methods.
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from tensorflow.python import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm # Fancy progress bars

from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



# Lets try on on all classes to see actual result on whole data.
# Get the image ids and corresponding breed labels.
y = df['enc_breed'].values
img_ids = df['id'].values


# Lets create the training, validation and test set from the image ids and breed labels.
id_train, id_test, y_train, y_test = train_test_split(img_ids, y, test_size=0.2, stratify = y, random_state=13)
id_train, id_val, y_train, y_val = train_test_split(id_train, y_train, test_size=0.2, stratify = y_train, random_state=13)

INPUT_SIZE = 299
# Create train, val and test data generators.
data_path = "./data/train"
BATCH = 32
train_data_gen = image_generator("inceptionresnet", data_path, id_train, y_train, batch_size = BATCH, height = INPUT_SIZE, width = INPUT_SIZE)
test_data_gen = image_generator("inceptionresnet", data_path, id_test, y_test, batch_size = BATCH, height = INPUT_SIZE, width = INPUT_SIZE)


inceptionresnet_feature_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling="avg")

# Geting features for train and validation dataset.
train_vgg_fs = inceptionresnet_feature_model.predict_generator(train_data_gen, steps = round(y_train.shape[0]/BATCH), verbose=1)
test_vgg_fs = inceptionresnet_feature_model.predict_generator(test_data_gen, steps = round(y_test.shape[0]/BATCH), verbose=1)

# Print obtained feature shape.
print('Xception training set features shape: {} size: {:,}'.format(train_vgg_fs.shape, train_vgg_fs.size))
print('Xception test set features shape: {} size: {:,}'.format(test_vgg_fs.shape, test_vgg_fs.size))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Set the parameters by cross-validation
param_grid = {
                'bootstrap': [True, False],
                'max_depth': [10, 20,],
                'max_features': ['auto',],
                'n_estimators': [200,]
                }

clf = GridSearchCV(RandomForestClassifier(), param_grid, cv = 3)
clf.fit(train_vgg_fs, y_train)

print("Best parameters:",clf.best_params_)
print("Best training score:",clf.best_score_)


# prediction results
predictions = clf.predict(test_vgg_fs)
print("Accuracy:", accuracy_score(y_test,predictions))
print(classification_report(y_test, predictions))


