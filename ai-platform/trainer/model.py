import argparse
import json
import os

from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import datetime
import functools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_dataset(x, y, batch_size, type='TRAIN'):
    
    #Load Pandas DF into tf dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(x), y.values))
    
    #Apply shuffle, batch and repeat to the training data. For evaluation we apply batch.
    if type == 'TRAIN': 
        dataset = dataset.shuffle(1000).batch(batch_size).repeat()
    else: 
        dataset = dataset.batch(32)
    
    # We take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.prefetch(1)
    
    return dataset

def create_dnn_model(INPUTS, NUMERIC_COLS, CATEGORICAL_COLS):
    
    #Define Input Features
    INPUTS = INPUTS
    
    #Set up feature columns list
    feature_columns = []
    feature_layer_inputs = {}
    
    for colname in INPUTS:
        feature_columns.append(tf.feature_column.numeric_column(colname))
        feature_layer_inputs[colname] = tf.keras.Input(shape=(1,), name=colname) 
        
    
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    dnn_input = feature_layer(feature_layer_inputs)
    
    #Layer One
    h1 = tf.keras.layers.Dense(32, activation='relu', name='h1')(dnn_input)
    
    #Layer Two
    h2 = tf.keras.layers.Dense(8, activation='relu', name='h2')(h1)
    
    #Model Output
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(h2)
    
    model = keras.Model(feature_layer_inputs, output)
    #model = keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=output)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model

def train_and_evaluate(args):
    
    x_train_data = args['x_train_data_paths']
    y_train_data = args['y_train_data_paths']

    x_test_data = 'gs://crazy-hippo-01/dataset/x_test.csv'
    y_test_data = 'gs://crazy-hippo-01/dataset/y_test.csv'
    
    x_val = args['x_eval_data_paths']
    y_val = args['y_eval_data_paths']
    
    
    X_train = pd.read_csv(tf.io.gfile.GFile(x_train_data))
    y_train = pd.read_csv(tf.io.gfile.GFile(y_train_data))
    
    X_test = pd.read_csv(tf.io.gfile.GFile(x_test_data))
    y_test = pd.read_csv(tf.io.gfile.GFile(y_test_data))
    
    x_val = pd.read_csv(tf.io.gfile.GFile(x_val))
    y_val = pd.read_csv(tf.io.gfile.GFile(y_val))

    FEATURES = X_train.columns
    NUMERIC_COLS = X_train.columns[2:]
    CATEGORICAL_COLS = X_train.columns[:2]
    
    #Create TF datasets
    train_dataset = create_dataset(X_train, y_train, args['train_batch_size'], 'TRAIN')
    eval_dataset = create_dataset(x_val, y_val, args['train_batch_size'], 'EVAL').take(args['num_eval_examples'])
    
    #Create Model Architecture
    func_model = create_dnn_model(FEATURES, NUMERIC_COLS, CATEGORICAL_COLS)

    NUM_TRAIN_EXAMPLES = len(X_train)
    steps_per_epoch = NUM_TRAIN_EXAMPLES // (args['train_batch_size'] * args['epochs'])

    #Train model
    history = func_model.fit(train_dataset,
         epochs= args['epochs'],
         verbose=1, 
         steps_per_epoch=steps_per_epoch, 
         validation_data=eval_dataset,
             )
    
    test_dataset = create_dataset(X_test, y_test, args['train_batch_size'], 'TEST')

    test_loss, test_acc = func_model.evaluate(test_dataset, steps=20)
    print('Test Loss:', test_loss, 'Test Accuracy:', test_acc)
    
    #export the model
    import shutil, os, datetime
    save_path = os.path.join(args['output_dir'], 'export/savedmodel')
    #func_model.save(save_path)
    
    tf.keras.models.save_model(func_model, save_path)
    print('Model Saved in', save_path)

    