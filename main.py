# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 19:27:53 2021

@author: User
"""

import os
from DogBreedClassify import DogBreedClassify

db_clf = DogBreedClassify("vgg")

train = False
test = False
predict = True

if train:
    response = db_clf.train()
    if response:
        print("Training successful.")
    else:
        print("Training failed.")
    
if test:
    response = db_clf.test()
    if response:
        print("Test successful.")
    else:
        print("Test failed.")
        
        
img_path = r"C:\Work\Self_task\Dog_Breed_Classification\data\train\00a338a92e4e7bf543340dc849230e75.jpg"

if predict:
    response = db_clf.predict("file",img_path)
    if response:
        print("Output:",response)
    else:
        print("Prediction failed.")
        
        
    