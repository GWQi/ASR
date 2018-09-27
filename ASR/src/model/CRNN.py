# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-20
# Email: hey_xiaoqi@163.com
# Filename: CRNN.py
# Description: construct CRNN network, cnn is composed of 1d residual network
# *************************************************

import tensorflow as tf

bottleneck_cnt = 0
def bottleneck_1d(inputs, kernel_size, reduce_dim, out_channels, dilation_rate, is_training, keep_prob):
  """
  construct standard bottleneck block
  param inputs : tf.Tensor, shape=[batches, max_steps, channels]
  param kernel_size : integer, kernel size along time axis
  param reduce_dim : integer, reduced dimension
  param out_channels : integer, out channels of this block
  param dilation_rate : integer, dilation rate
  param is_training : bool or tf.bool, weather in training phase
  param keep_prob : float, keep probability when drop out
  """
  shape = inputs.get_shape().as_list()

  global bottleneck_cnt
  bottleneck_cnt += 1
  with tf.name_scope("bottleneck/%d" % bottleneck_cnt):
    # reduce dimension
    reduce_conv = tf.layers.conv1d(inputs, filters=reduce_dim, kernel_size=1, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
    # reduce_bn = tf.layers.batch_normalization(reduce_conv, training=is_training)
    reduce_relu = tf.nn.relu(reduce_conv)

    # atrous convolution
    atrous_conv = tf.layers.conv1d(reduce_relu, filters=out_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
    atrous_bn = tf.layers.batch_normalization(atrous_conv, training=is_training)
    atrous_relu = tf.nn.relu(atrous_bn)

    # restore the dimension
    restore_conv = tf.layers.conv1d(atrous_relu, filters=out_channels, kernel_size=1, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
    # restore_bn = tf.layers.batch_normalization(restore_conv, training=is_training)
    restore_relu = tf.nn.relu(restore_conv)

    # shortcuts
    if shape[-1] == out_channels:
      # identity shortcuts
      outputs = inputs + restore_relu

    else:
      # projection shortcuts
      outputs = tf.layers.conv1d(inputs, filters=out_channels, kernel_size=1, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer()) + restore_relu


    # dropout
    outputs = tf.layers.dropout(outputs, rate=1-keep_prob, training=is_training)

    return outputs

dilation_rate_list = [1, 2, 4, 8, 16, 32]
def bigblock_1d(inputs, num_block, kernel_size, reduce_dim, out_channels, is_training, keep_prob):
  """
  construct a big block using bottleneck block
  param inputs : tf.Tensor, shape=[batches, max_steps, channels]
  param num_block : integer, number of small bottleneck block in this big block
  param kernel_size : integer, kernel size along time axis
  param reduce_dim : integer, reduced dimension
  param out_channels : integer, out channels of this block
  param is_training : bool or tf.bool, weather in training phase
  param keep_prob : float, keep probability when drop out
  """
  shape = inputs.get_shape().as_list()

  initial_inputs = tf.identity(inputs)

  global dilation_rate_list
  for i in list(range(num_block)):
    inputs = bottleneck_1d(inputs, kernel_size, reduce_dim, out_channels, dilation_rate_list[i], is_training, keep_prob)

  # shortcuts
  if shape[-1] == out_channels:
    # identity shortcuts
    outputs = initial_inputs + inputs
  else:
    # projection shortcuts
    outputs = tf.layers.conv1d(initial_inputs, filters=out_channels, kernel_size=1, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer()) + inputs

  return outputs

def CRNN(inputs, is_training, NUM_LABELS):
  """
  features is extracted by python_speech_features, so the features.shape=[max_steps, N_orders]
  inputs.shape=[batches, max_steps, N_orders]
  so the inputs is a 3D placehoder, which shape=[batches, max_steps, N_orders]
  params:
    inputs : 3D tensor(placeholder), inputs.shape=[batches, max_steps, N_orders]
    is_training : bool or tf.bool, indicate weather in training phase
    NUM_LABELS : int, number of labels in lexical
  """
  shape = tf.shape(inputs)

  with tf.name_scope("expand"):
    inputs_conv = tf.layers.conv1d(inputs, filters = 128, kernel_size=7, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    inputs_relu = tf.nn.relu(inputs_conv)

  # bigblock 1
  with tf.name_scope("bigblock/1"):
    bigblock_1_outputs = bigblock_1d(inputs_relu, num_block=5, kernel_size=3, reduce_dim=64, out_channels=128, is_training=is_training, keep_prob=0.8)

  # # bigblock 2
  # with tf.name_scope("bigblock/2"):
  #   bigblock_2_outputs = bigblock_1d(bigblock_1_outputs, num_block=5, kernel_size=3, reduce_dim=64, out_channels=128, is_training=is_training, keep_prob=0.8)

  # # bigblock 3
  # with tf.name_scope("bigblock/3"):
  #   bigblock_3_outputs = bigblock_1d(bigblock_2_outputs, num_block=5, kernel_size=3, reduce_dim=64, out_channels=128, is_training=is_training, keep_prob=0.8)

  # # bigblock 4
  # with tf.name_scope("bigblock/4"):
  #   bigblock_4_outputs = bigblock_1d(bigblock_3_outputs, num_block=5, kernel_size=3, reduce_dim=64, out_channels=128, is_training=is_training, keep_prob=0.8)

  # FNN
  with tf.name_scope("logit"):
    FNN = tf.layers.conv1d(bigblock_1_outputs, filters=128, kernel_size=1, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
    FNN_bn = tf.layers.batch_normalization(FNN, training=is_training)
    FNN_relu = tf.nn.relu(FNN_bn)

    logit = tf.layers.conv1d(FNN_relu, filters=NUM_LABELS+1, kernel_size=1, kernel_initializer=tf.contrib.layers.xavier_initializer())

  return logit