# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-20
# Email: hey_xiaoqi@163.com
# Filename: resnet.py
# Description: construct deep residual(with dilated) network
# *************************************************

import tensorflow as tf

def expand_dim_by_1dconv(inputs, out_channels, is_training):
  """
  expand feature dimension and expand one extra dimension for inputs to be [batches, max_steps, 1, channels]
  param inputs : 3D tensor(placeholder), inputs.shape=[batches, max_steps, N_orders]
  param out_channels : int, chennels after expanding
  param is_training : bool or tf.bool, weather in training phase
  return outputs : expanded inputs of resnet
  """
  # expand one extra dimension, inputs.shape=[batches, max_steps, 1, N_orders]
  inputs = tf.expand_dims(inputs, axis=2)
  with tf.name_scope("expand"):
    expand_1_conv = tf.layers.conv2d(inputs, int(out_channels/2), kernel_size=3, padding='same', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
    expand_1_bn = tf.layers.batch_normalization(expand_1_conv, training=is_training)
    expand_1_relu = tf.nn.relu(expand_1_bn)

    expand_2_conv = tf.layers.conv2d(expand_1_relu, out_channels, kernel_size=3, padding='same', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
    expand_2_bn = tf.layers.batch_normalization(expand_2_conv, training=is_training)
    expand_2_relu = tf.nn.relu(expand_2_bn)
    return expand_2_relu

def conv2d_bn_relu_layer(inputs, filters, kernel_size,
                         conv_strides, is_training,
                         dilation_rate=(1, 1), conv_pad='same'):
  """
  construct a convlution layers, with batch normalization, relu activation layer
  params:
    inputs : 4D tensor inputs, inputs.shape=[batches, hight, width, channel]
    filters : int, number of filters/channels of outputs
    kernel_size : int or list/tuple of 2 integers, (hight, width)
    conv_strides : int or list/tuple of 2 integers, (hight, width)
    is_training : python bool or tf.bool(normally placeholder), weather in training phase
    dilation_rate : dilation rate along hight and width
    conv_pad : convolutional padding, 'same' or 'valid'
  """
  # convolution layer
  conv_outputs = tf.layers.conv2d(inputs, filters, kernel_size,
                                  strides=conv_strides, padding=conv_pad,
                                  dilation_rate=dilation_rate, use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
  # batch normalization
  batch_norm_outputs = tf.layers.batch_normalization(conv_outputs, training=is_training)
  # relu activation
  relu_outputs = tf.nn.relu(batch_norm_outputs)
  return relu_outputs

resblock_cnt = 0
def resblock_1d(inputs, kernel_size, out_channels, is_training, keep_prob):
  """
  construct standard resblock, in 1D convolution
  param inputs : tf.Tensor, shape=[batches, max_steps, 1, channels]
  param kernel_sizel : integer, kernel size along time axis
  param out_channels : out channels of this block
  param is_training : bool or tf.bool, weather in training phase
  param keep_prob : keep probability when drop out
  """
  shape = inputs.get_shape().as_list()

  global resblock_cnt
  resblock_cnt += 1
  with tf.name_scope("resblock/%d" % resblock_cnt):
    conv_layer_1 = conv2d_bn_relu_layer(inputs, out_channels, (kernel_size, 1), 1, is_training)
    conv_layer_2 = conv2d_bn_relu_layer(conv_layer_1, out_channels, (kernel_size, 1), 1, is_training)

    if shape[-1] == out_channels:
      # identity shortcuts
      outputs = inputs + conv_layer_2
    else:
      # projection
      outputs = tf.layers.conv2d(inputs, out_channels, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), use_bias=False)

    # dropout
    outputs = tf.layers.dropout(outputs, rate=1-keep_prob, training=is_training )
    return outputs

def bigblock_1d(inputs, num_resblock, kernel_size, out_channels, is_training, keep_prob):
  """
  construct bigblock contain several standard resblocks, in 1D convolution
  param inputs : tf.Tensor, shape=[batches, max_steps, 1, channels]
  param num_resblock : int, resblocks in this big block
  param kernel_sizel : integer, kernel size along time axis
  param out_channels : out channels of this big block
  param is_training : bool or tf.bool, weather in training phase
  param keep_prob : keep probability when drop out
  """
  
  shape = inputs.get_shape().as_list()

  inputs_initial = tf.identity(inputs)
  for i in list(range(num_resblock)):
    inputs = resblock_1d(inputs, kernel_size, out_channels, is_training, keep_prob)

  if shape[-1] == out_channels:
    # identity shortcuts
    outputs = inputs_initial + inputs
  else:
    # projection
    outputs = tf.layers.conv2d(inputs_initial, out_channels, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), use_bias=False) + inputs
  # no dropout after bigblock
  return outputs


def resnet(inputs, is_training, NUM_LABELS):
  """
  features is extracted by python_speech_features, so the features.shape=[max_steps, N_orders]
  inputs.shape=[batches, max_steps, N_orders]
  so the inputs is a 3D placehoder, which shape=[batches, max_steps, N_orders]
  params:
    inputs : 3D tensor(placeholder), inputs.shape=[batches, max_steps, N_orders]
    is_training : bool or tf.bool, indicate weather in training phase
    NUM_LABELS : int, number of labels in lexical
  """

  # feature expanding to 64
  resnet_inputs = expand_dim_by_1dconv(inputs, 64, is_training)

  # bigblock 1
  with tf.name_scope("bigblock/1"):
    bigblock_1_outputs = bigblock_1d(inputs=resnet_inputs, num_resblock=5, kernel_size=7,
                                     out_channels=64, is_training=is_training, keep_prob=0.8)

  # bigblock 2
  with tf.name_scope("bigblock/2"):
    bigblock_2_outputs = bigblock_1d(inputs=bigblock_1_outputs, num_resblock=5, kernel_size=7,
                                     out_channels=64, is_training=is_training, keep_prob=0.8)

  # bigblock 3
  with tf.name_scope("bigblock/3"):
    bigblock_3_outputs = bigblock_1d(inputs=bigblock_2_outputs, num_resblock=5, kernel_size=7,
                                     out_channels=128, is_training=is_training, keep_prob=0.8)

  # bigblock 4
  with tf.name_scope("bigblock/4"):
    bigblock_4_outputs = bigblock_1d(inputs=bigblock_3_outputs, num_resblock=5, kernel_size=7,
                                     out_channels=128, is_training=is_training, keep_prob=0.8)
  # bigblock 5
  with tf.name_scope("bigblock/5"):
    bigblock_5_outputs = bigblock_1d(inputs=bigblock_4_outputs, num_resblock=5, kernel_size=7,
                                     out_channels=256, is_training=is_training, keep_prob=0.8)

  with tf.name_scope("logits"):
    FNN = tf.layers.conv2d(inputs=bigblock_5_outputs, filters=256, kernel_size=1, use_bias=False,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())
    FNN_bn = tf.layers.batch_normalization(FNN, training=is_training)
    FNN_relu = tf.nn.relu(FNN_bn)
    logits = tf.layers.conv2d(inputs=FNN_relu, filters=NUM_LABELS+1, kernel_size=1,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.squeeze(logits, axis=2)
  # logits.shape=[batches, max_stepsize, NUM_LABELS+1]
  return logits
