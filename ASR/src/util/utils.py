# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: audio.py
# Description: some useful tools
# *************************************************

import numpy as np

# ASR
def pad_sequence(data):
  """
  pad zeros for each array in data
  params:
    data : list of np.ndarrays [[Tframs, Norders], ...]
  return:
    pad_data : np.ndarray, padded data
    stepsizes : valid step size for each sample
  """
  # valid step size for each sample

  stepsizes = np.asarray([len(seq) for seq in data], dtype = np.int32)
  # number of samples
  nsamples = stepsizes.size
  # max step size of those samples
  maxlength = stepsizes.max()

  fea_shape = data[0].shape[1:]

  pad_data = np.zeros((nsamples, maxlength) + fea_shape, dtype=data[0].dtype)

  for i in np.arange(nsamples):
    pad_data[i, :stepsizes[i]] = data[i]

  return pad_data, stepsizes


def dense2sparse(labels):
  """
  transform dense indexes labels into sparse format
  params:
    labels : list of lists of integers
  return:
    indices: 
    values:
    shape:
  """
  indices = []
  values = []
  for n, label in enumerate(labels):
    # for m, label_idx in enumerate(label):
    #   if label_idx != 0:
    #     indices.append((n, m))
    #     values.append(label_idx)
    indices.extend(zip([n]*len(label), list(range(len(label)))))
    values.extend(label)

  indices = np.asarray(indices, dtype=np.int32)
  values = np.asarray(values, dtype=np.int32)
  shape = np.asarray([len(labels), indices.max(axis=0)[1]+1])

  return indices, values, shape
