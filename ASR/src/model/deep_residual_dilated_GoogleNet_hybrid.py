# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-20
# Email: hey_xiaoqi@163.com
# Filename: deep_residual_dilated_GoogleNet_hybrid.py
# Description: construct deep residual dilated GoogleNet hybrid network
# *************************************************

import tensorflow as tf

# ************************************************
# * configuration of feature dimension expanding *
# ************************************************
feature_dim = 128

def expand_dim_by_2dconv(inputs, is_training):
  """
  param inputs : 3D tensor(placeholder), inputs.shape=[batches, T_frames, N_orders]
  param is_training : bool or tf.bool, weather in training phase
  """
  global feature_dim

  shape = tf.shape(inputs)
  # expand_layers_inputs.shape = [batches, T_frames, N_orders, 1]
  expand_layers_inputs = tf.expand_dims(inputs, axis=-1)
  with tf.name_scope("expand/1"):
    layer_1_conv = tf.layers.conv2d(expand_layers_inputs, filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    layer_1_bn = tf.layers.batch_normalization(layer_1_conv, training=is_training)
    layer_1_relu = tf.nn.relu(layer_1_bn)
  with tf.name_scope("expand/2"):
    layer_2_normal_conv = tf.layers.conv2d(layer_1_relu, filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    layer_2_normal_bn = tf.layers.batch_normalization(layer_2_normal_conv, training=is_training)
    layer_2_normal_relu = tf.nn.relu(layer_2_normal_bn)
    
    layer_2_atrous_conv = tf.layers.conv2d(layer_1_relu, filters=16, kernel_size=3, strides=1, padding='same', dilation_rate=(2, 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
    layer_2_atrous_bn = tf.layers.batch_normalization(layer_2_atrous_conv, training=is_training)
    layer_2_atrous_relu = tf.nn.relu(layer_2_atrous_bn)

    layer_2_concat = tf.concat([layer_2_normal_relu, layer_2_atrous_relu], axis=-1)

  # reshape layers 2 concat to [batches, T_frames, -1]
  layer_2 = tf.reshape(layer_2_concat, [shape[0], shape[1], 640])
  # expand one dim after frame axis to apply reduce dimension of feature
  layer_2 = tf.expand_dims(layer_2, axis=2)
  outputs_conv = tf.layers.conv2d(layer_2, filters=feature_dim, kernel_size=1, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
  outputs_bn = tf.layers.batch_normalization(outputs_conv, training=is_training)
  outputs_relu = tf.nn.relu(outputs_bn)

  # outputs_relu.shape=[batches, T_frame, 1, feature_dim]
  return outputs_relu


# ***************************************
# * configuration of one residual block *
# ***************************************
resblock_kernel_size = 3
resblock_dilation_num = 3
resblock_dilation_rate = [1, 2, 4]
resblock_filters = [int(feature_dim/2), int(feature_dim/4), int(feature_dim/4)]
res_googleNet_atrous_1d_cnt = 0

def resblock_googleNet_atrous_1d(inputs, is_training):
  """
  this residual block combine
  param inputs : 4D tensor, shape=[batches, T_frames, 1, feature_dim]
  param is_training : bool or tf.bool, weather in training phase
  """

  global resblock_kernel_size
  global resblock_dilation_num
  global resblock_dilation_rate
  global resblock_filters
  global res_googleNet_atrous_1d_cnt
  # count the block number
  res_googleNet_atrous_1d_cnt += 1

  with tf.name_scope("resblock/%d" % res_googleNet_atrous_1d_cnt):
    # atrous convolution layers
    atrous_layer_outputs = []
    for i in list(range(resblock_dilation_num)):
      atrous_conv = tf.layers.conv2d(inputs, filters=resblock_filters[i],
                                     kernel_size=[resblock_kernel_size, 1],
                                     strides=[1, 1], padding='same',
                                     dilation_rate=[resblock_dilation_rate[i], 1], use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
      atrous_bn = tf.layers.batch_normalization(atrous_conv, training=is_training)
      atrous_relu = tf.nn.relu(atrous_bn)
      atrous_layer_outputs.append(atrous_relu)

    # concatenate all the atrous layer ouputs, atrous_layer_concat.shape=[batches, T_frames, 1, feature_dim]
    atrous_layer_concat = tf.concat(atrous_layer_outputs, axis=-1)

    # 1x1 kernel to aggregate the different context scale information
    atrous_layer_aggregate = tf.layers.conv2d(atrous_layer_concat,
                                              filters=atrous_layer_concat.get_shape().as_list()[-1],
                                              kernel_size=[1, 1], strides=[1, 1], padding='same',
                                              use_bias=False,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    atrous_aggregate_bn = tf.layers.batch_normalization(atrous_layer_aggregate, training=is_training)
    atrous_aggregate_relu = tf.nn.relu(atrous_aggregate_bn)
    # identity short cuts
    block_outputs = inputs + atrous_aggregate_relu

  return block_outputs

# **********************************
# * configuration of one big block *
# **********************************
resblock_num_in_bigblock = 1

def bigblock_googleNet_atrous_1d(inputs, is_training):
  """
  construct a big "block" using resblock_googleNet_atrous_1d(...)
  param inputs : 4D tensor, shape=[batches, T_frames, 1, feature_dim]
  is_training : bool or tf.bool, weather in training phase
  """
  global resblock_num_in_bigblock

  outputs = resblock_googleNet_atrous_1d(inputs, is_training)
  for i in list(range(resblock_num_in_bigblock-1)):
    outputs = resblock_googleNet_atrous_1d(outputs, is_training)

  # identity short cuts
  return outputs + inputs


bigblock_num = 1
def DRDG_1dconv(inputs, stepsizes, is_training, NUM_LABELS):
  """
  features is extracted by python_speech_features, so the features.shape=[T_frames, N_orders]
  so the inputs is a 3D placehoder, which shape=[batches, T_frames, N_orders]
  params:
    inputs : 3D tensor(placeholder), inputs.shape=[batches, T_frames, N_orders]
    stepsizes : valid stepsize of each training instance
    is_training : bool or tf.bool, indicate weather in training phase
    NUM_LABELS : int, number of labels in lexical
  """

  shape = tf.shape(inputs)
  batches, max_stepsize = shape[0], shape[1]
  # 2D-CNN layer to expand the feature dim
  network_inputs = expand_dim_by_2dconv(inputs, is_training)

  # deep residual dilation googlenet hybrif network
  global bigblock_num
  for i in list(range(bigblock_num)):
    with tf.name_scope("bigblock/%d" % (i+1)):
      network_inputs = bigblock_googleNet_atrous_1d(network_inputs, is_training)

  # dense prediction(semantic segmentation/classification)
  dense_inputs = network_inputs
  with tf.name_scope("dense/1"):
    dense_1_conv = tf.layers.conv2d(dense_inputs, filters=dense_inputs.get_shape().as_list()[-1],
                                    kernel_size=[5, 1], strides=[1, 1], padding="same",
                                    use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
    dense_1_bn = tf.layers.batch_normalization(dense_1_conv, training=is_training)
    dese_1_relu = tf.nn.relu(dense_1_bn)
  
  # logits
  with tf.name_scope("dense/2"):
    logits = tf.layers.conv2d(dense_inputs, filters=NUM_LABELS+1,
                              kernel_size=[5, 1], strides=[1, 1], padding="same",
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.squeeze(logits, axis=2)

  # logits.shape=[batches, max_stepsize, NUM_LABELS+1]
  return logits