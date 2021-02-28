
import tensorflow as tf
import tensorflow_transform as tft
#import features
import importlib

#importlib.reload(features)

CATEGORICAL_FEATURES = ['workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'native_country']

NUMERIC_FEATURES = [ 'age', 'education_num', 'capital_gain', 'capital_loss',
    'hours_per_week', 'fnlwgt']

INPUTS = ['workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'native_country', 'age', 'education_num', 'capital_gain', 'capital_loss',
    'hours_per_week', 'fnlwgt']

LABEL_KEY = 'income_bracket'

#NUMERIC_FEATURES = features.NUMERIC_FEATURES
#CATEGORICAL_FEATURES = features.CATEGORICAL_FEATURES
#LABEL_KEY = features.LABEL_KEY
#VOCAB_SIZE = binary_constants.VOCAB_SIZE
#OOV_SIZE = binary_constants.OOV_SIZE
#TRANSFORMED_NAME = binary_constants.transformed_name



def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
      Args:
        inputs: map from feature keys to raw not-yet-transformed features.
      Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = inputs.copy()
    
    #Scale numeric columns to have range [0,1] - if Tensors.
    for key in NUMERIC_FEATURES:
        #outputs[key] = tft.scale_to_0_1(inputs[key])
        #outputs[transformed_name(key)] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))
        outputs[key] = tft.scale_to_0_1(_fill_in_missing(outputs[key]))
    
    # For all categorical columns except the label column, we generate a
    # vocabulary but do not modify the feature.  This vocabulary is instead
    # used in the trainer, by means of a feature column, to convert the feature
    # from a string to an integer id.
    
    for key in CATEGORICAL_FEATURES:
        outputs[key] = tft.compute_and_apply_vocabulary(
                                            _fill_in_missing(outputs[key]),
                                            #top_k=VOCAB_SIZE,
                                            #num_oov_buckets=OOV_SIZE
                                            )
    outputs[LABEL_KEY] = _fill_in_missing(outputs[LABEL_KEY])
    
    return outputs

def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
        x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
          in the second dimension.
        Returns:
        A rank 1 tensor where missing values of `x` have been filled in.
    """     
    
    if not isinstance(x, tf.sparse.SparseTensor):
        return x
    
    default_value = '' if x.dtype == tf.string else 0
    
    return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)