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

from . import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--x_train_data_paths',
        help = 'GCS or local path to training data',
        #type = str,
        #required = True
    )
    parser.add_argument(
        '--y_train_data_paths',
        help = 'GCS or local path to training data',
        type = str,
        #required = True
    )
    parser.add_argument(
        '--train_batch_size',
        help = 'Batch size for training steps',
        type = int,
        default = 32
    )
    parser.add_argument(
        '--epochs',
        help = 'Steps to run the training job for',
        type = int
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Number of steps to run evalution for at each checkpoint',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--num_eval_examples',
        help = 'Number of steps to run evalution for at each checkpoint',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--x_eval_data_paths',
        help = 'GCS or local path to evaluation data',
        type = str,
        #required = True
    )
    parser.add_argument(
        '--y_eval_data_paths',
        help = 'GCS or local path to evaluation data',
        type = str,
        #required = True
    )
    # Training arguments
    parser.add_argument(
        '--hidden_units',
        help = 'List of hidden layer sizes to use for DNN feature columns',
        nargs = '+',
        type = int,
        default = [32, 32, 4]
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        #required = True
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )

    # Eval arguments
    parser.add_argument(
        '--eval_delay_secs',
        help = 'How long to wait before running first evaluation',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--throttle_secs',
        help = 'Seconds between evaluations',
        default = 300,
        type = int
    )

    args = parser.parse_args()
    arguments = args.__dict__
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


    # Unused args provided by service
    #arguments.pop('job_dir', None)
    #arguments.pop('job-dir', None)

    #output_dir = arguments['output_dir']
    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    
    #output_dir = os.path.join(
    #    output_dir,
    #    json.loads(
    #        os.environ.get('TF_CONFIG', '{}')
    #    ).get('task', {}).get('trail', '')
    #)

    # Run the training job
    model.train_and_evaluate(arguments)