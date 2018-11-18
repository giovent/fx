import tensorflow as tf

import utils

ACTIVATIONS = {'Sigmoid': tf.nn.sigmoid}

def activation_function(tensor, function_type):
  if function_type in ACTIVATIONS:
    return ACTIVATIONS[function_type](tensor)
  utils.log(type='Warning', msg='Couldn\'t find activation function: {type}'.format(type=function_type),
            during='tf model building')

def LSTM(input, input_length, output_dim=256, rnn_units=128):
  input_ = tf.unstack(input, input_length, 1)
  with tf.name_scope('RNN_Layer'):
    with tf.variable_scope('lstm'):
      lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_units, forget_bias=1.0)
      outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, input_, dtype=tf.float32)
    rnn_output = outputs  # [-1] for last output
  return rnn_output

def fully_connected_layer(input, output_dim, activation_type='Sigmoid', input_dropout=1, name="FCLayer"):
  input_dim = int(input.shape[-1])
  with tf.name_scope('Layer'):
    W = tf.Variable(tf.random_normal([input_dim, output_dim], 0.0, 0.1), name='W')
    b = tf.Variable(tf.random_normal([output_dim], 0.0, 0.1), name='Bias')
    return activation_function(tf.matmul(input, W) + b, function_type=activation_type)