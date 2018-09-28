# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:2018-9-26
# Email: hey_xiaoqi@163.com
# Filename: DNN_LSTM.py
# Description: construct dnn+lstm model for digit recognization
# *************************************************

import tensorflow as tf

def DNN_LSTM(inputs, is_training, NUM_LABELS):
  """
  params:
    inputs : 3D tensor(placeholder), inputs.shape=[batches, max_steps, N_orders]
    is_training : bool or tf.bool, indicate weather in training phase
    NUM_LABELS : int, number of labels in lexical
  """
  

