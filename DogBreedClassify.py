# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 20:11:43 2021

@author: User
"""

# Importing all required modules and methods.
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
from PIL import Image
import io

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
# Computing class weights.
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras

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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input, decode_predictions
# from keras.applications.resnet50 import ResNet50
# from keras.applications import xception
# from keras.applications import inception_v3

from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Model
#from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg16 import decode_predictions as decode_vgg
from tensorflow.python.keras.applications.vgg16 import preprocess_input as vgg_process
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input as inres_process
from tensorflow.python.keras.applications.inception_resnet_v2 import decode_predictions as decode_inres
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import numpy as np
from tensorflow.keras import mixed_precision


class DogBreedClassify:
    def __init__(self, model_type):
        """
        Initialize class instance.
        """
        self.model_type = model_type
        
    def read_id_label_mapping(self):
        """
        Read the data that maps image id to dog breed label.
        """
        
        path_to_data = './data'
        self.df = pd.read_csv(os.path.join(path_to_data,'labels.csv'))


    def encode_labels(self):
        """
        Encode the dog breed labels in the data mapping.
        """
        
        # Encode the breed name in data csv.
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.df['breed'])
        self.df['enc_breed'] = self.label_encoder.transform(self.df['breed'])
        np.save("encoded_classes.npy",self.label_encoder.classes_)
        
        
    def get_train_test_data(self):
        """
        Divide data into train, validation and split.
        Further calculate class weights for training.
        """
        # Get the image ids and corresponding breed labels.
        y = self.df['enc_breed'].values
        img_ids = self.df['id'].values


        # Lets create the training, validation and test set from the image ids
        # and breed labels.
        id_train, id_test, y_train, y_test = train_test_split(img_ids, y, 
                                                              test_size=0.2, 
                                                              stratify = y, 
                                                              random_state=13)
        id_train, id_val, y_train, y_val = train_test_split(id_train, y_train, 
                                                            test_size=0.2, 
                                                            stratify = y_train,
                                                            random_state=13)
        
        
        # Computing class weights.
        class_weights = compute_class_weight('balanced',
                                            np.unique(np.ravel(y_train,order='C')),
                                            np.ravel(y_train,order='C'))
        
        self.class_weights = {i : class_weights[i] for i in range(len(class_weights))}
        
        
        # We will modify the dataframe a bit to enable working with image data generators.
        # Adding .jpg at the end of each value in 'id' column.
        self.df["id"] = self.df["id"] + ".jpg"
        
        self.id_train = [img_id + ".jpg" for img_id in id_train]
        self.id_val = [img_id + ".jpg" for img_id in id_val]
        self.id_test = [img_id + ".jpg" for img_id in id_test]
        self.y_test = y_test
        
        # Add another column, label_set in df that denotes whether id is in train or test set.
        self.df["label_set"] = self.df["id"].apply(lambda val : "train" \
                                                   if (val in self.id_train) \
                                                       else ("val" if (val in self.id_val) \
                                                             else ("test" if (val in self.id_test) \
                                                                   else None)))
            
            
    def define_vgg_generators(self):
        """
        Define image data generators with data augmentation parameters for VGG16.
        """
        # Define the Image Data Generators for augmentation.
        train_datagen = ImageDataGenerator(rotation_range=90, 
                                             brightness_range=[0.1, 0.7],
                                             width_shift_range=0.5, 
                                             height_shift_range=0.5,
                                             horizontal_flip=True, 
                                             vertical_flip=True,
                                             validation_split=0.15,
                                             preprocessing_function = vgg_process)
        
        test_datagen = ImageDataGenerator(preprocessing_function = vgg_process)
        
        
        self.train_generator = train_datagen.flow_from_dataframe(
                            dataframe=self.df[self.df["label_set"] == "train"], 
                            directory="./data/train", x_col="id", y_col="breed",
                            class_mode="categorical", subset = "training", 
                            target_size=(self.INPUT_SIZE,self.INPUT_SIZE), 
                            batch_size=self.BATCH, shuffle=True, seed = 13)
        
        self.val_generator = train_datagen.flow_from_dataframe(
                            dataframe=self.df[self.df["label_set"] == "val"], 
                            directory="./data/train", x_col="id", y_col="breed",
                            class_mode="categorical", subset = "validation", 
                            target_size=(self.INPUT_SIZE,self.INPUT_SIZE), 
                            batch_size=self.BATCH, shuffle=True, seed = 13)
        
        self.test_generator = test_datagen.flow_from_dataframe(
                            dataframe=self.df[self.df["label_set"] == "test"], 
                            directory="./data/train", x_col="id", y_col=None, 
                            class_mode=None, target_size=(self.INPUT_SIZE,
                                                          self.INPUT_SIZE), 
                            batch_size=1, shuffle=False, seed = 13)


    def get_vgg_model(self):
        """
        Take pre-trained VGG16 model, remove the top layer and add custom layers.
        Returns compiled model.
        """
        # Now we take the VGG16 pretrained model and remove the top layer. We will add our own layer.
        # Pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        # Use half precision floating points in network.
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        
        input_shape = (self.INPUT_SIZE, self.INPUT_SIZE, 3)
        n_classes = int(self.df["breed"].nunique())
        # print("\n\n",n_classes,"\n\n")
        optim_1 = tf.keras.optimizers.Adam(lr=0.001)
        
        conv_base = VGG16(include_top=False,
                         weights='imagenet', 
                         input_shape=input_shape)
        
        
        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        top_model = conv_base.output
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(1028, activation='relu',dtype='float32')(top_model)
        top_model = Dense(512, activation='relu',dtype='float32')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(n_classes, activation='softmax',dtype='float32')(top_model)
        
        # Group the convolutional base and new fully-connected layers into a Model object.
        vgg_model = Model(inputs=conv_base.input, outputs=output_layer)

        # Compiles the model for training.
        vgg_model.compile(optimizer=optim_1, 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return vgg_model
        
        
    def compile_model(self):
        """
        Based on choice, compile vgg16 or inception-resnet50 model.
        """
        if self.model_type in ("vgg","vgg16","VGG","VGG16"):
            self.BATCH = 8
            self.INPUT_SIZE = 224
            self.model = self.get_vgg_model
        else:
            print("Invalid model choice.")
            self.model = None
            
        
    def train(self):
        """
        Train the compiled model.
        """
        # Read id - label mapping data.
        self.read_id_label_mapping()
        
        # Encode labels.
        self.encode_labels()
        
        # Get train, validation and test datasets.
        self.get_train_test_data()
        
        # Define data generators and model.
        if self.model_type in ("vgg","vgg16","VGG","VGG16"):
            self.BATCH = 8
            self.INPUT_SIZE = 224
            self.define_vgg_generators()
            self.model = self.get_vgg_model()
        else:
            print("Invalid model choice.")
            return False
            
        
        # Compute number of training and validation steps.
        n_steps = self.train_generator.samples // self.BATCH
        n_val_steps = self.val_generator.samples // self.BATCH
        n_epochs = 10
        
        # Lets create checkpoint to save the model and define early stopping conditions.
        # ModelCheckpoint callback - save best weights
        tl_checkpoint_1 = ModelCheckpoint(filepath='dbc_{}_best.hdf5'\
                                          .format(self.model_type),
                                          save_best_only=True,
                                          verbose=2)
        
        
        self.model_history = self.model.fit(self.train_generator,
                                    batch_size=self.BATCH,
                                    epochs=n_epochs,
                                    validation_data=self.val_generator,
                                    steps_per_epoch=n_steps,
                                    validation_steps=n_val_steps,
                                    callbacks=[tl_checkpoint_1],
                                    class_weight=self.class_weights,
                                    verbose=1)
        
        return True
    
    
    def test(self):
        """
        Get test predictions and metrics from test data generated.
        This uses the test data generators created previously.
        """
        # Read id - label mapping data.
        self.read_id_label_mapping()
        
        # Encode labels.
        self.encode_labels()
        
        # Get train, validation and test datasets.
        self.get_train_test_data()
        
        # Define data generators and model.
        if self.model_type in ("vgg","vgg16","VGG","VGG16"):
            self.BATCH = 8
            self.INPUT_SIZE = 224
            self.define_vgg_generators()
            self.model = self.get_vgg_model()
        else:
            print("Invalid model choice.")
            return False
        
        try:
            # Load best trained model.
            self.model.load_weights('dbc_{}_best.hdf5'.format(self.model_type))
            
            # Generate predictions
            true_classes = self.y_test
            class_indices = self.train_generator.class_indices
            class_indices = dict((v,k) for k,v in class_indices.items())
            
            
            preds = self.model.predict(self.test_generator)
            pred_classes = np.argmax(preds, axis=1)
            
            acc = accuracy_score(true_classes, pred_classes)
            print("{0} Model Accuracy: {1:.2f}%"\
                  .format(self.model_type, acc * 100))
                
            return True
        except:
            traceback.print_exc()
            return False
            
            
    def predict(self, input_type, image_path):
        """
        Get prediction for image sent from model based on model chosen.
        """
        # Read id - label mapping data.
        self.read_id_label_mapping()
        
        # Encode labels.
        self.encode_labels()
        
        # Load label encoder classes.
        encoder = LabelEncoder()
        encoder.classes_ = np.load('encoded_classes.npy', allow_pickle=True)
        
        # Define data generators and model.
        if self.model_type in ("vgg","vgg16","VGG","VGG16"):
            self.BATCH = 8
            self.INPUT_SIZE = 224
            self.model = self.get_vgg_model()
            target_size = (224, 224)
            
            # Load best trained model.
            self.model.load_weights('dbc_{}_best.hdf5'.format(self.model_type))
            
            # Load image.
            if input_type == "file":
                image = load_img(image_path, target_size=target_size)
            else:
                image = Image.open(io.BytesIO(image_path))
                image = image.resize(target_size, Image.NEAREST)
            # Convert the image pixels to a numpy array
            image = img_to_array(image)
            # Reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # Prepare the image for the VGG model
            image = vgg_process(image)
            # Predict the probability across all output classes
            yhat = self.model.predict(image)
            # Convert the probabilities to class labels
            label = np.argmax(yhat, axis=1)
            
            # Return the actual label.
            return encoder.inverse_transform(label)[0]
            
        else:
            print("Invalid model choice.")
            return False