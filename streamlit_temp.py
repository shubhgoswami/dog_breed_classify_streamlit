# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 22:20:35 2021

@author: User
"""


import streamlit as st 
from PIL import Image
from DogBreedClassify import DogBreedClassify
from io import StringIO


st.title("Dog Breed Identifier")

model_type = st.radio("Choose base model...", ("VGG16","Inception-Resnet50"))
if model_type == "VGG16":
    model_type = "vgg"
    
db_clf = DogBreedClassify(model_type)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize([int(0.5 * s) for s in image.size], Image.NEAREST)
    st.image(image, caption='Uploaded Image.', use_column_width=False)
    st.write("")
    st.write("Identifying...")
    # image_bytes = StringIO(uploaded_file.getvalue().decode("utf-8"))
    label = db_clf.predict("bytes",uploaded_file.getvalue())
    st.write('%s' % (label))