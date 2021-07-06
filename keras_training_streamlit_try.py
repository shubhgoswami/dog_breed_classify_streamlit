# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 21:30:54 2021

@author: User
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import streamlit as st


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

st.title("Keras Training UI")

inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

train_steps_per_epoch = len(x_train) //batch_size

# Prepare the validation dataset.
# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

if st.button("Train"):
    st.write("Starting training with 2 epochs...")
    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        st.write("Epoch {}".format(epoch+1))
        start_time = time.time()
        progress_bar = st.progress(0.0)
        percent_complete = 0
    
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train)
    
            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * 64))
                percent_complete = ((step/train_steps_per_epoch))
                progress_bar.progress(percent_complete)
        
        progress_bar.progress(1.0)
    
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
    
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
    
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            test_step(x_batch_val, y_batch_val)
    
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        st.write("Duration : {0:.2f}s, Training acc. : {1:.4f}, Validation acc.:{2:.4f}"\
                 .format((time.time() - start_time),float(train_acc),float(val_acc)))