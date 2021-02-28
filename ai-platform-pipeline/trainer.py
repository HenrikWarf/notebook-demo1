from typing import List, Text

import os
import absl
import datetime
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow.keras import layers

from tfx.components.trainer.executor import TrainerFnArgs

_INPUTS = ['workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'native_country', 'age', 'education_num', 'capital_gain', 'capital_loss',
    'hours_per_week', 'fnlwgt']

_LABEL_KEY = 'income_bracket'


def transformed_name(key):
    return key + '_xf'


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')

#In the run_fn above, a serving signature is needed when exporting the trained model 
#so that model can take raw examples for prediction. A typical serving function would look like this:

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop('income_bracket')
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
        """Generates features and label for tuning/training.

        Args:
        file_pattern: List of paths or patterns of input tfrecord files.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of returned
          dataset to combine in a single batch

        Returns:
            A dataset that contains (features, indices) tuple where features is a
              dictionary of Tensors, and indices is a single Tensor of label indices.
        """
        transformed_feature_spec = (
          tf_transform_output.transformed_feature_spec().copy())

        dataset = tf.data.experimental.make_batched_features_dataset(
          file_pattern=file_pattern,
          batch_size=batch_size,
          features=transformed_feature_spec,
          reader=_gzip_reader_fn,
          label_key=_LABEL_KEY
          #label_key=transformed_name(_LABEL_KEY))
         )
        
        return dataset


def create_dnn_model(INPUTS):
    
    #Define Input Features
    INPUTS = INPUTS
    
    #Set up feature columns list
    feature_columns = []
    feature_layer_inputs = {}

    
    for colname in INPUTS:
        feature_columns.append(tf.feature_column.numeric_column(colname))
        feature_layer_inputs[colname] = tf.keras.Input(shape=(1,), name=colname) 
    
    #Inputs to model
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
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model




def run_fn(fn_args: TrainerFnArgs):
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

    TRAIN_BATCH_SIZE = 32
    NUM_TRAIN_EXAMPLES = 10000
    EPOCHS = 5  
    NUM_EVAL_EXAMPLES = 100 
    
    func_model = create_dnn_model(_INPUTS)
    print('Your model has been built.... Again')
    
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    
    func_model.fit(
             train_dataset,
             epochs= EPOCHS,
             verbose=1, 
             steps_per_epoch=fn_args.train_steps, 
             validation_steps=fn_args.eval_steps,
             validation_data=eval_dataset,
            )
    
    signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(func_model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
      }
    
    func_model.save(fn_args.serving_model_dir, 
                    save_format='tf', 
                    signatures=signatures
                   )