# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-26
# Email: hey_xiaoqi@163.com
# Filename: thefuck.py
# Description: construct deep residual(with dilated gated) network
# *************************************************
import tensorflow as tf
# conv1d_layer
conv1d_index = 0
def conv1d_layer(input_tensor, size, dim, activation, scale, bias, is_training):
  global conv1d_index
  with tf.variable_scope('conv1d_' + str(conv1d_index)):
    W = tf.get_variable('W', (size, input_tensor.get_shape().as_list()[-1], dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
    if bias:
      b = tf.get_variable('b', [dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
    out = tf.nn.conv1d(input_tensor, W, stride=1, padding='SAME') + (b if bias else 0)
    if not bias:
      beta = tf.get_variable('beta', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
      gamma = tf.get_variable('gamma', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
      mean_running = tf.get_variable('mean', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
      variance_running = tf.get_variable('variance', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
      mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)))
      def update_running_stat():
        decay = 0.99
        update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), variance_running.assign(variance_running * decay + variance * (1 - decay))]
        with tf.control_dependencies(update_op):
          return tf.identity(mean), tf.identity(variance)
      m, v = tf.cond(is_training, update_running_stat, lambda: (mean_running, variance_running))
      out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)
    if activation == 'tanh':
      out = tf.nn.tanh(out)
    if activation == 'sigmoid':
      out = tf.nn.sigmoid(out)
 
    conv1d_index += 1
    return out

# aconv1d_layer
aconv1d_index = 0
def aconv1d_layer(input_tensor, size, rate, activation, scale, bias, is_training):
  global aconv1d_index
  with tf.variable_scope('aconv1d_' + str(aconv1d_index)):
    shape = input_tensor.get_shape().as_list()
    W = tf.get_variable('W', (1, size, shape[-1], shape[-1]), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
    if bias:
      b = tf.get_variable('b', [shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
    out = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, dim=1), W, rate=rate, padding='SAME')
    out = tf.squeeze(out, [1])
    if not bias:
      beta = tf.get_variable('beta', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
      gamma = tf.get_variable('gamma', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
      mean_running = tf.get_variable('mean', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
      variance_running = tf.get_variable('variance', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
      mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)) )
      def update_running_stat():
        decay = 0.99
        update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), variance_running.assign(variance_running * decay + variance * (1 - decay))]
        with tf.control_dependencies(update_op):
          return tf.identity(mean), tf.identity(variance)
      m, v = tf.cond(is_training, update_running_stat, lambda: (mean_running, variance_running))
      out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)
    if activation == 'tanh':
      out = tf.nn.tanh(out)
    if activation == 'sigmoid':
      out = tf.nn.sigmoid(out)
 
    aconv1d_index += 1
    return out

n_dim = 128
def thefuck(inputs, is_training, NUM_LABELS):
  global n_dim
  out = conv1d_layer(input_tensor=inputs, size=1, dim=n_dim, activation='tanh', scale=0.14, bias=False, is_training=is_training)
  # skip connections
  def residual_block(input_sensor, size, rate):
    conv_filter = aconv1d_layer(input_sensor, size=size, rate=rate, activation='tanh', scale=0.03, bias=False, is_training=is_training)
    conv_gate = aconv1d_layer(input_sensor, size=size, rate=rate,  activation='sigmoid', scale=0.03, bias=False, is_training=is_training)
    out = conv_filter * conv_gate
    out = conv1d_layer(out, size=1, dim=n_dim, activation='tanh', scale=0.08, bias=False, is_training=is_training)
    return out + input_sensor, out

  skip = 0
  for _ in list(range(3)):
    for r in [1, 2, 4, 8]:
      out, s = residual_block(out, size=7, rate=r)
      skip += s
 
  logit = conv1d_layer(skip, size=1, dim=skip.get_shape().as_list()[-1], activation='tanh', scale=0.08, bias=False, is_training=is_training)
  logit = conv1d_layer(logit, size=1, dim=NUM_LABELS+1, activation=None, scale=0.04, bias=True, is_training=is_training)
 
  return logit