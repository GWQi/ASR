# -*- coding: utf-8 -*-

import os
import sys
import logging
import tensorflow as tf

ASR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ASR_ROOT)

from ASR.src.base.JDDIterator import JDDIterator

JDD_MODEL_DIR = os.path.join(ASR_ROOT, 'ASR/ckpt/jdd/resnet')
logfile = os.path.join(JDD_MODEL_DIR, 'log.txt')
checkpoint_prefix = os.path.join(JDD_MODEL_DIR, 'ckpt')

DATA_ROOT = '/home/guwenqi/Documents/jdd_mandarin/feature'
LABELS_PATH = '/home/guwenqi/Documents/jdd_mandarin/label.txt'

def conv_bn_relu_layer(inputs_layer, weight, stride, is_training):
  """
  a helper function to conv, batch normalization, relu activation
  params:
    inputs_layer : 4D tensor
    weight : filter
    stride : int ,stride along with frequence axis
    is_training : weather training or validation, training phase if True
  return:
    outputs of this layer
  """

  # convolution layer
  conv_layer = tf.nn.conv2d(inputs_layer, weight, strides=[1, stride, 1, 1], padding='SAME')
  # batch normalizatiom
  batch_norm = tf.layers.batch_normalization(conv_layer, training=is_training)
  # activation
  outputs_layer = tf.nn.relu(batch_norm)

  return outputs_layer


def residual_block(inputs_layer, out_channels, is_training, ithblock):
  """
  define a residual block
  params:
    inputs_layer : inputs of this block, 4D
    out_channels : out channels of this block
    is_training : weather training or validation, training phase if True
    ithblock : int, ith block of this resnet
  return:
    outputs_layer : outputs of this block, 4D tensor
  """
  in_channels = inputs_layer.get_shape().as_list()[-1]
  # weather increase channel after this block
  increase_channel = False
  if in_channels * 2 == out_channels:
    increase_channel = True
    stride = 2
  elif in_channels == out_channels:
    increase_channel = False
    stride = 1
  else:
    raise ValueError("Out channel and in channel don't match at {}'th block!!!".format(ithblock))

  with tf.variable_scope("resnet/{}_block".format(ithblock)):
    weight_1 = tf.get_variable('w_1', shape=[3, 3, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer())
    conv_1 = conv_bn_relu_layer(inputs_layer, weight_1, stride, is_training)
    weight_2 = tf.get_variable('w_2', shape=[3, 3, out_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer())
    conv_2 = conv_bn_relu_layer(conv_1, weight_2, 1, is_training)
  
    if increase_channel:
      # if the input channel and output channel of this block are different, use 1x1 filter to pad
      shortcuts_weight = tf.get_variable('w_3', shape=[1, 1, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer())
      shortcuts = tf.nn.conv2d(inputs_layer, shortcuts_weight, strides=[1, stride, 1, 1], padding='SAME')
    
    else:
      # if in channel and out channel are same, Identity map shortcuts
      shortcuts = inputs_layer

  outputs_layer = conv_2 + shortcuts
  return outputs_layer


def CRNN(NUM_LABELS):
  """
  resnet
  """
  # layers 
  layers = {}
  # inputs.shape=[batch, time_axis, fea_axis]
  inputs = tf.placeholder(tf.float32, shape=[None, None, None])
  targets = tf.sparse_placeholder(tf.int32)
  stepsizes = tf.placeholder(tf.int32, shape=[None])
  is_training = tf.placeholder(tf.bool)

  layers['inputs'] = inputs
  layers['targets'] = targets
  layers['stepsizes'] = stepsizes
  layers['is_training'] = is_training

  inputs_shape = tf.shape(inputs)
  batches, max_timestep = inputs_shape[0], inputs_shape[1]

  # transpose inputs from [batches, time_axis, fea_axis] to [batches, fea_axis, time_axis]
  # rank of inputs is 3, insert one dim it to 4
  inputs = tf.transpose(inputs, [0, 2, 1])
  inputs = tf.expand_dims(inputs, -1)

  with tf.variable_scope('resnet/inputs'):
    w = tf.get_variable('w', shape=[5, 5, 1, 16], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', shape=[16], initializer=tf.zeros_initializer())
    resnet_inputs = tf.nn.relu(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding='SAME') + b)
  resnet_inputs = tf.nn.max_pool(resnet_inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding='SAME')
  
  # construct deep resdual network
  # for out_channel in argv['channels']:
  channels = [16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128]
  for i in list(range(len(channels))):
    resnet_inputs = residual_block(resnet_inputs, channels[i], is_training, i+1)

  # transpose resnet outputs from [batch, feature_axis, frame_axis, channel] to [batch, frame_axis, feature_axis, channel]
  resnet_outputs = tf.transpose(resnet_inputs, perm=[0, 2, 1, 3])
  # reshape resnet outputs to [batch, frame_axis, feature]
  num_units = 512
  resnet_outputs = tf.reshape(resnet_inputs, [batches, max_timestep, num_units])
  # num_units = resnet_outputs.get_shape().as_list()[-1]

  # BiRNN network
  # define cell type
  cell = tf.contrib.rnn.GRUCell

  # first BiRNN layer
  with tf.variable_scope('birnn/1'):
    cell_fw = cell(num_units)
    cell_bw = cell(num_units)
    (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, resnet_outputs, sequence_length=stepsizes, dtype=tf.float32)
    # batch normalization
    outputs_fw_bn = tf.layers.batch_normalization(outputs_fw, training=is_training)
    outputs_bw_bn = tf.layers.batch_normalization(outputs_bw, training=is_training)

    w_fw = tf.get_variable('w_fw', shape=[num_units, num_units])
    w_bw = tf.get_variable('w_bw', shape=[num_units, num_units])
    outputs_fw_bn = tf.reshape(outputs_fw_bn, [-1, num_units])
    outputs_bw_bn = tf.reshape(outputs_bw_bn, [-1, num_units])
    outputs_birnn_1 = tf.reshape(tf.matmul(outputs_fw_bn, w_fw)+tf.matmul(outputs_bw_bn, w_bw), [batches, max_timestep, num_units])

  with tf.variable_scope('birnn/2'):
    cell_fw = cell(num_units)
    cell_bw = cell(num_units)
    (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, outputs_birnn_1, sequence_length=stepsizes, dtype=tf.float32)
    # batch normalization
    outputs_fw_bn = tf.layers.batch_normalization(outputs_fw, training=is_training)
    outputs_bw_bn = tf.layers.batch_normalization(outputs_bw, training=is_training)

    outputs_birnn_2 = tf.concat((outputs_fw_bn, outputs_bw_bn), 2)

  # FNN
  FNN_inputs = tf.reshape(outputs_birnn_2, [-1, num_units*2])
  with tf.variable_scope('fnn'):
    w = tf.get_variable('w', shape=[num_units*2, NUM_LABELS+1])
    b = tf.get_variable('b', shape=[NUM_LABELS+1])
    FNN_outputs = tf.nn.relu(tf.matmul(FNN_inputs, w) + b)

  logits = tf.reshape(FNN_outputs, [batches, max_timestep, NUM_LABELS+1])

  # logits.shape = [batches, max_timestep, NUM_LABELS+1]
  return logits, layers


def train():
  """
  train network
  """
  # jdd iterator initialization
  jdditer = JDDIterator()
  try:
    jdditer.load(os.path.join(JDD_MODEL_DIR, 'JDDIterator.ckpt'))
  except IOError:
    jdditer.configure(DATA_ROOT, LABELS_PATH)

  # logger configuration
  logging.basicConfig(filename=logfile,
                      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                      filemode='a',
                      level=logging.DEBUG)
  logger = logging.getLogger('Training procession')

  g_train = tf.Graph()
  with g_train.as_default():

    # logits.shape = [batches, max_timestep, NUM_LABELS]
    logits, layers = CRNN(len(jdditer.lexical))
    # transpose logits to time major format
    logits = tf.transpose(logits, perm=[1, 0, 2])
    cost = tf.reduce_mean(tf.nn.ctc_loss(layers['targets'], logits, layers['stepsizes']))
    learning_rate = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, layers['stepsizes'])

    label_err_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), layers['targets']))
    saver = tf.train.Saver(max_to_keep=5)

  with tf.Session(graph=g_train) as sess:
    try:
      saver.restore(sess, checkpoint_prefix)
    except:
      tf.global_variables_initializer().run()

    print("Begin train:")
    while True:
      data, stepsizes, targets, epoch_done = jdditer.next_batch()
      if data.size != 0:
        cost_, _ = sess.run([cost, optimizer],
                             feed_dict={layers['inputs'] : data,
                                       layers['targets'] : targets,
                                       layers['stepsizes'] : stepsizes,
                                       layers['is_training'] : True,
                                       learning_rate : 0.001 * 0.9**(jdditer.kth_epoch-1)
                                        })
        logger.info("{}'th epoch, {}'th batch, cost value: {}".format(jdditer.kth_epoch, jdditer.ith_batch, cost_))
        
        if jdditer.ith_batch % 20 == 0:
          saver.save(sess, checkpoint_prefix)
          jdditer.save(os.path.join(JDD_MODEL_DIR, 'JDDIterator.ckpt'))
          logger.info("{}'th epoch, {}'th batch save model and iterator.".format(jdditer.kth_epoch, jdditer.ith_batch))

        if epoch_done:
          # save model
          saver.save(sess, checkpoint_prefix)
          jdditer.save(os.path.join(JDD_MODEL_DIR, 'JDDIterator.ckpt'))
          logger.info("{}'th epoch, {}'th batch save model and iterator.".format(jdditer.kth_epoch, jdditer.ith_batch))
          
          # validation
          count = int(len(jdditer.val_list) / jdditer.batch_size) + 1
          ler_all = 0
          for i in list(range(count)):
            data, stepsizes, targets = jdditer.fetch_data(i*jdditer.batch_size, (i+1)*jdditer.batch_size)
            ler = sess.run([label_err_rate],
                             feed_dict={layers['inputs'] : data,
                                       layers['targets'] : targets,
                                       layers['stepsizes'] : stepsizes,
                                       layers['is_training'] : False
                                        })
            ler_all += ler
          logger.info("{}'th epoch, label error rate on validation set: {}".format(jdditer.kth_epoch, ler_all/count))

      else:
        logger.info("Train finished!")
        break

        
if __name__ == '__main__':
  train()
