# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:2018-9-26
# Email: hey_xiaoqi@163.com
# Filename: JDDDigitIterator.py
# Description: JDD Digit data iterator class defination
# *************************************************

import os
import pickle
import numpy as np

from io import open
from random import shuffle
from collections import Counter

from ASR.src.base.DataIterator import DataIterator
from ASR.src.util.utils import pad_sequence, dense2sparse



class JDDDigitIterator(DataIterator):

  def __init__(self):
    """
    params:
      root : string, data root
      path : string, labels file path
    """
    super(JDDDigitIterator, self).__init__()
    # self.train_list = [[filename1, [label1, label2, ...]], ... ]
    # self.val_list = [[filename1, [label1, label2, ...]], ... ]
    # self.lexical = {'character1' : index1, 'character2' : index2, ...}
    self.lexical = {}
    # self.lexical_inverse = {index1 : 'character1', index2 : 'character2', ...}
    self.lexical_inverse = {}

    self.batch_size = 30
    self.num_epoch = 100

  # @override
  def next_batch(self):
    """
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
      filepath, datalabels = self.train_list[index]
      fea = np.load(os.path.join(self.data_root, filepath+'.npy'))
      # translate character labels into index labels
      dataindexes = [self.lexical.get(datalabel, 0) for datalabel in datalabels]

      data.append(fea)
      targets.append(dataindexes)

    # pad each sample to max length of this batch
    data, stepsizes = pad_sequence(data)
    # translate dense labels into sparse format to feed tf graph
    targets = dense2sparse(targets)

    return data, targets, stepsizes, epoch_done

  # @override
  def fetch_data(self, start, end, dataname='val'):
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
    for filepath, datalabels in datalist[start : end]:
      fea = np.load(os.path.join(self.data_root, filepath+'.npy'))
      # translate character labels into index labels
      dataindexes = [self.lexical.get(datalabel, 0) for datalabel in datalabels]

      data.append(fea)
      targets.append(dataindexes)

    # pad each sample to max length of this batch
    data, stepsizes = pad_sequence(data)
    # translate dense labels into sparse format to feed tf graph
    targets = dense2sparse(targets)

    return data, stepsizes, targets

  # @override
  def configure(self, root, path):
    """
    configuraion of jdd data iterator
    """
    # initializethe batch size and number epoch
    self.data_root = root

    # validation data partion
    val_partion = 0.1
    words = []

    with open(path, 'r', encoding='utf-8') as f:
      for aline in f.readlines():
        filename, labels = aline.strip().split()
        labels = list(labels)
        words.extend(labels)

        self.data_list.append([filename, labels])

    # lexical and reverse lexical configuration
    words_counter = Counter(words)
    words_counter = sorted(words_counter.items(), key=lambda x: x[-1], reverse=True)
    lexical, _ = zip(*words_counter)
    self.lexical = dict(zip(lexical, list(range(0, len(lexical)))))
    self.lexical_inverse = dict(zip(self.lexical.values(), self.lexical.keys()))

    # data_list, train_list, val_list, val_list configuration
    val_size = int(len(self.data_list) * val_partion)
    # shuffle the original data list to generate train list and val/test list
    shuffle(self.data_list)
    self.train_list = self.data_list[0:-val_size]
    self.val_list = self.data_list[-val_size:]
    self.test_list = list(self.val_list)

    # generate train indexes
    self.train_indexes = list(range(len(self.train_list)))

  # @override
  def save(self, ckpt_path):
    """
    save this iterator
    params:
      ckpt_path : checkpoint path
    """
    with open(ckpt_path, 'wb') as f:
      pickle.dump(self, f)


  # @override
  def load(self, ckpt_path):
    """
    restore this iterator
    params:
      ckpt_path : checkpoint path
    """
    with open(ckpt_path, 'rb') as f:
      ckpt = pickle.load(f)
      self.__dict__ = ckpt.__dict__

  
  def set_rootpath(self, path):
    """
    used to set wav root path
    param : path, string, data root path
    """
    self.data_root = path
    



    
    
