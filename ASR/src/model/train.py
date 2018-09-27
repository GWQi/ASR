# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-9-21
# Email: hey_xiaoqi@163.com
# Filename: train.py
# Description: this script is used to train the network
# *************************************************
import os
import sys
import logging
import tensorflow as tf

# append ASR directory path to the system path
ASR_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ASR_PROJECT_ROOT)

from ASR.src.base import fparam
from ASR.src.base.AishellIterator import AishellIterator
from ASR.src.base.JDDIterator import JDDIterator
from ASR.src.base.THCHSIterator import THCHSIterator
from ASR.src.base.JDDDigitIterator import JDDDigitIterator

from deep_residual_dilated_GoogleNet_hybrid import DRDG_1dconv
from resnet import resnet
from thefuck import thefuck
from CRNN import CRNN


def create_flags():
  tf.app.flags.DEFINE_float  ("grad_clip",         -1.0,                 "gradient clip value")
  tf.app.flags.DEFINE_string ("MODEL_ROOT",       "../../model/jdd",        "directory where to save checkpoint")
  tf.app.flags.DEFINE_string ("CKPT_PREFIX",      "../../model/ckpt",   "tensorflow checkpoint prefix")
  tf.app.flags.DEFINE_string ("DATA_ROOT",        "/home/guwenqi/Documents/jdd_new/data/feature/mfcc_no_norm",                   "data root directory path")
  tf.app.flags.DEFINE_string ("TRANSCRIPTS_PATH", "/home/guwenqi/Documents/jdd_new/data/label.txt",                   "transcripts file path")
  tf.app.flags.DEFINE_string ("logfile",          "./log.txt",          "log file path")
  # tf.app.flags.DEFINE_integer("NUM_LABELS",       4333,                    "number of words in Aishell lexical")
  # tf.app.flags.DEFINE_integer("NUM_LABELS",       2026,                    "number of words in JDD mandarin lexical")
  # tf.app.flags.DEFINE_integer("NUM_LABELS",       2883,                    "number of words in JDD mandarin lexical")
  tf.app.flags.DEFINE_integer("NUM_LABELS",         12,                    "number of words in JDD digit lexical")

def train():
  # flags initialization
  create_flags()
  FLAGS = tf.app.flags.FLAGS

  # iterator initialization
  dataiter = JDDDigitIterator()
  try:
    dataiter.load(os.path.join(FLAGS.MODEL_ROOT, "JDDDigitIterator.ckpt"))
    dataiter.set_rootpath(FLAGS.DATA_ROOT)
  except:
    dataiter.configure(FLAGS.DATA_ROOT, FLAGS.TRANSCRIPTS_PATH)

  # logger configuration
  logging.basicConfig(filename=FLAGS.logfile, format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                      filemode='a', level=logging.DEBUG)
  logger = logging.getLogger()

  # ******************
  # * draw the graph *
  # ******************
  # inputs of the network,
  # features is extracted by python_speech_features, so the features.shape=[T_frames, N_orders]
  inputs = tf.placeholder(tf.float32, shape=[None, None, 20])
  targets = tf.sparse_placeholder(tf.int32)
  stepsizes = tf.placeholder(tf.int32, shape=[None])
  learning_rate = tf.placeholder(tf.float32)
  is_training = tf.placeholder(tf.bool)

  # logits.shape=[batches, max_timestep, NUM_LABELS+1]
  logits = CRNN(inputs, is_training, FLAGS.NUM_LABELS+1)
  # transpose logits to time-major format
  logits = tf.transpose(logits, [1, 0, 2])
  # ctc loss
  loss = tf.reduce_mean(tf.nn.ctc_loss(targets, logits, stepsizes, time_major=True))
  # decode the logits
  decoded, probs = tf.nn.ctc_beam_search_decoder(logits, stepsizes)
  # compute the edit distance between predicts and ground truth, ie. label error rate
  ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

  # create the optimizer, because the batch normalization moving average update
  # operation relies no-gradient updating, so we need update the mean and std manually
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    if FLAGS.grad_clip < 0:
      train_op = optimizer.minimize(loss)
    else:
      # # wrap optimizer with tf's estimator decorator, this is the simpliest method
      # optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=FLAGS.grad_clip)
      # train_op = optimizer.minimize(loss)
      # compute gradients and apply grad clip for each grads
      gradients, variables = zip(*optimizer.compute_gradients(loss))
      # grad clip
      gradients = [None if gradient is None else tf.clip_by_norm(gradient, FLAGS.grad_clip) for gradient in gradients]
      train_op = optimizer.apply_gradients(zip(gradients, variables))

  # create a saver
  saver = tf.train.Saver(max_to_keep=10)


  # *************************
  # * session initialization*
  # *************************
  with tf.Session() as sess:
    # restore the last checkpoint
    try:
      with open(os.path.join(FLAGS.MODEL_ROOT, "checkpoint"), 'r') as f:
        last_checkpoint = f.readline().strip().split()[-1].strip('"')
        saver.restore(sess, last_checkpoint)
    except:
      tf.global_variables_initializer().run()

    # Begain training
    print("Begain training!")

    # print average loss and average accuracy every 100 batch
    count = 0
    loss_all = 0
    ler_all = 0

    while True:
      # fetch data
      data, targets_, stepsizes_, epoch_done = dataiter.next_batch()

      if data is None:
        # the last epoch is done, save current model
        break

      else:
        loss_, ler_, _ = sess.run([loss, ler, train_op],
                                  feed_dict={
                                  inputs         : data,
                                  targets        : targets_,
                                  stepsizes      : stepsizes_,
                                  learning_rate  : 0.001 * 0.96**dataiter.kth_epoch,
                                  is_training    : True
                                  })
        print("Epoch: %-2d, Batch: %-5d, Loss: %-5f." %\
              (dataiter.kth_epoch, dataiter.ith_batch, loss_))
        count += 1
        loss_all += loss_
        ler_all += ler_

      if dataiter.ith_batch % 100 == 0:
        # compute average loss and average accuracy every 100 batch
        loss_ave = loss_all / count
        ler_ave = ler_all / count
        # reset loss_all, ler_all, count
        loss_all = 0
        ler_all = 0
        count = 0

        print("Epoch: %-2d, Batch: %-5d, Ave Loss: %-5f, Ave LER: %-5f." %\
              (dataiter.kth_epoch, dataiter.ith_batch, loss_ave, ler_ave))

        if dataiter.ith_batch % 1000 == 0 or epoch_done:
          # do validation every 1000 batch or epoch_done
          val_batch_num = int(len(dataiter.val_list) / dataiter.batch_size) + 1
          val_ler_all = 0

          for i in list(range(val_batch_num)):
            val_data, val_targets, val_stepsizes = dataiter.fetch_data(i*dataiter.batch_size,
                                                                       (i+1)*dataiter.batch_size,
                                                                       dataset='val')
            val_ler_ = sess.run([loss, ler],
                                feed_dict={
                                inputs         : val_data,
                                targets        : val_targets,
                                stepsizes      : val_stepsizes,
                                learning_rate  : 0.001 * 0.96**dataiter.kth_epoch,
                                is_training    : False
                                })
            val_ler_all += val_ler_

          # save model 
          saver.save(sess, FLAGS.CKPT_PREFIX+"-%d-%d" % (dataiter.kth_epoch, dataiter.ith_batch))
          dataiter.save(os.path.join(FLAGS.MODEL_ROOT, ''))
          logger.info("Epoch: %-2d, Batch: %-5d, Validation LER: %-5f, Model Saved!" %
                      (dataiter.kth_epoch, dataiter.ith_batch, val_ler_all/val_batch_num))


if __name__ == "__main__":
  train()