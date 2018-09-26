# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:2018-9-26
# Email: hey_xiaoqi@163.com
# Filename: THCHSIterator.py
# Description: Aishell dataset data iterator class defination
# *************************************************

import os
import numpy as np
from io import open
from random import shuffle
from collections import Counter

from ASR.src.base.DataIterator import DataIterator
from ASR.src.base import fparam
from ASR.src.util.audio import extractMFCC
from ASR.src.util.utils import pad_sequence, dense2sparse

class THCHSIterator(DataIterator):

  def __init__(self):
    super(THCHSIterator, self).__init__()

    # self.data_list = [[path, u"labels"], ...]
    # self.lexical = {'character1' : index1, 'character2' : index2, ...}
    self.lexical = {}
    # self.lexical_inverse = {index1 : 'character1', index2 : 'character2', ...}
    self.lexical_inverse = {}

    self.batch_size = 16


  def set_rootpath(self, path):
    """
    used to set wav root path
    param : path, string, data root path
    """
    self.data_root = path

  def configure(self, data_root, transcripts):
    """
    used for data/train/dev/test list configuration
    param : root, string, data root path
    param : transcripts, string, path where store transcripts for all wav file
    """
    # set data root
    if data_root.endswith('/'):
      data_root = os.path.split(data_root)[0]
    self.data_root = data_root

    # get filename -> labels key-value dict, and collect all words
    words = u""
    file_labels_dict = {}
    with open(transcripts, 'r', encoding='utf-8') as f:
      for aline in f.readlines():
        # sample of aline : BAC009S0914W0294     也  在  所  不  惜  的  最后
        aline_split = aline.strip().split()
        file, labels = aline_split[0], u"".join(aline_split[1:])
        file_labels_dict[file] = labels
        words += labels

        # construct train/val/test dala list
        self.data_list.append([file, labels])
        if file.startswith("train"):
          self.train_list.append([file, labels])
        if file.startswith("dev"):
          self.val_list.append([file, labels])
        if file.startswith("test"):
          self.test_list.append([file, labels])


    # get word -> index lexical dicct and index -> word inverse lexical dict
    words_counter = Counter(words)
    words_freq = sorted(words_counter.items(), key=lambda x: x[-1], reverse=True)
    words, freq = zip(*words_freq)
    self.lexical = dict(zip(words, tuple(range(len(words)))))
    self.lexical_inverse = dict(zip(self.lexical.values(), self.lexical.keys()))

    # train indexes
    self.train_indexes = list(range(len(self.train_list)))

  def next_batch(self):
    """
    fetch next batch training data
    return:
      data  : list of feature arrays, np.ndarrays, has padded zeros
      stepsizes: list of int, valid length of each np.ndarray in data
      targets: np.sparse tensor of label
      epoch_done: bool, indicate one epoch done
    """
    # flag, indicate one epoch done
    epoch_done = False
    # fetched data
    data = []
    targets = []
    # advance iterator
    self.ith_batch += 1
    # one epoch is done
    if (self.ith_batch-1) * self.batch_size >= len(self.train_list):
      self.ith_batch = 1
      self.kth_epoch += 1
      epoch_done = True
      # weather all epoch done
      if self.kth_epoch > self.num_epoch:
        return None, None, None, True
      shuffle(self.train_indexes)

    # fetch data and labels
    for index in self.train_indexes[(self.ith_batch-1)*self.batch_size : self.ith_batch*self.batch_size]:
      filepath, labels = self.train_list[index]
      # extract features
      # fea = extractMFCC(os.path.join(self.data_root, filepath),
      #                   fparam.MFCC_ORDER, frame_length=fparam.MFCC_FRAME_LENGTH,
      #                   frame_shift=fparam.MFCC_FRAME_SHIFT)
      
      # load features
      fea = np.load(os.path.join(self.data_root, filepath+".npy"))
      # translate labels into indexes
      labelidxes = [self.lexical.get(label, 0) for label in labels]

      data.append(fea)
      targets.append(labelidxes)

    # pad each sample to max length of this batch
    data, stepsizes = pad_sequence(data)
    # translate dense labels into sparse format to feed tf graph
    targets = dense2sparse(targets)

    # samples in data has shape=[T_frames, N_mfcc]
    return data, targets, stepsizes, epoch_done

  def fetch_data(self, start, end, dataset='val'):
    """
    featch data
    params:
      dataname: string, if 'val', fetch validation data, else if 'test', fetch test data
      start : int, start file index of data
      end : int, end file index of data
    """
    datalist = []
    if dataname == 'val':
      datalist = self.val_list
    elif dataname == 'test':
      datalist = self.test_list
    else:
      raise ValueError('you just can fetch validation data and test data')

    # fetched data
    data = []
    targets = []
    for filepath, labels in datalist[start : end]:
      fea = extractMFCC(os.path.join(self.data_root, filepath),
                        fparam.MFCC_ORDER, frame_length=fparam.MFCC_FRAME_LENGTH,
                        frame_shift=fparam.MFCC_FRAME_SHIFT)
      # translate character labels into index labels
      labelidxes = [self.lexical.get(label, 0) for label in labels]

      data.append(fea)
      targets.append(labelidxes)

    # pad each sample to max length of this batch
    data, stepsizes = pad_sequence(data)
    # translate dense labels into sparse format to feed tf graph
    targets = dense2sparse(targets)

    return data, targets, stepsizes

  def save(self, ckpt_path):
    """
    save this iterator
    params:
      ckpt_path : checkpoint path
    """
    with open(ckpt_path, 'wb') as f:
      pickle.dump(self, f)


  def load(self, ckpt_path):
    """
    restore this iterator
    params:
      ckpt_path : checkpoint path
    """
    with open(ckpt_path, 'rb') as f:
      ckpt = pickle.load(f)
      self.__dict__ = ckpt.__dict__

# import sys
# sys.path.append('/media/gwq/Seagate Expansion Drive/ASR')
# from AishellIterator import AishellIterator
# aiter = AishellIterator()
# aiter.configure('/media/gwq/Seagate Expansion Drive/fengxindata/compressed/aishell/aishell_dataset_one/data_aishell/wav', '/media/gwq/Seagate Expansion Drive/fengxindata/compressed/aishell/aishell_dataset_one/data_aishell/transcript/aishell_transcript_v0.8.txt')
