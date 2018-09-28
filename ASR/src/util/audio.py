# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: audio.py
# Description: extract feature from audio files
# *************************************************


import python_speech_features
from scipy.io import wavfile
import numpy as np
import math

def extractMFCC(filepath, numcep, frame_length=0.025,
                frame_shift=0.01, n_stride=1, n_context=0,
                delta=False, double_delta=False):
  """
  frame_length, frame_shift in seconds.
  """
  fs, data = wavfile.read(filepath)

  # transform steoro to mono
  if data.ndim == 2:
    data = data.mean(axis=-1)

  features = python_speech_features.mfcc(data, fs, frame_length, frame_shift, numcep, nfilt=64, nfft=int(math.ceil(frame_length*fs)))

  if delta:
    delta_features = compute_delta(features)
    if double_delta:
      double_delta_features = compute_delta(delta_features)

  if delta:
    features = np.concatenate((features, delta_features), axis=-1)
    if double_delta:
      features = np.concatenate((features, double_delta_features), axis=-1)

  train_inputs = features[::n_stride]
  if n_context != 0:
    num_strides = features.shape[0]

    zeros_context = np.zeros([n_context, numcep], dtype=features.dtype)
    features = np.concatenate([zeros_context, features, zeros_context])

    window_size = 2*n_context+1

    train_inputs = np.lib.stride_tricks.as_strided(
              features,
              (num_strides, window_size, numcep),
              (features.strides[0], features.strides[0], features.strides[1]),
              writeable=False)

    train_inputs = train_inputs.reshape([num_strides, -1])

  # train_inputs.shape = [numframes, numcep]
  return train_inputs


def extractMBE(filepath, numband, frame_length=0.025, frame_shift=0.01, n_stride=1, n_context=0):
  """
  frame_length, frame_shift in seconds.
  """
  fs, data = wavfile.read(filepath)

  # transform steoro to mono
  if data.ndim == 2:
    data = data.mean(axis=-1)

  features, _ = python_speech_features.fbank(data, fs, frame_length, frame_shift, nfilt=numband, nfft=int(math.ceil(frame_length*fs)))

  train_inputs = features[::n_stride]
  if n_context != 0:
    num_strides = features.shape[0]

    zeros_context = np.zeros([n_context, numband], dtype=features.dtype)
    features = np.concatenate([zeros_context, features, zeros_context])

    window_size = 2*n_context+1

    train_inputs = np.lib.stride_tricks.as_strided(
              features,
              (num_strides, window_size, numband),
              (features.strides[0], features.strides[0], features.strides[1]),
              writeable=False)

    train_inputs = train_inputs.reshape([num_strides, -1])

  # train_inputs.shape = [numframes, numband]
  return train_inputs


# reference: sidekit
def compute_delta(features, win=3, method='filter', filt=numpy.array([.25, .5, .25, 0, -.25, -.5, -.25])):
  """
  features is a 2D-ndarray  each row of features is a a frame
    
  param features: the feature frames to compute the delta coefficients
  param win: parameter that set the length of the computation window. The size of the window is (win x 2) + 1
  param method: method used to compute the delta coefficients can be diff or filter
  param filt: definition of the filter to use in "filter" mode, default one is similar to SPRO4:  filt=numpy.array([.2, .1, 0, -.1, -.2])
        
  return: the delta coefficients computed on the original features.
  """
  # First and last features are appended to the begining and the end of the 
  # stream to avoid border effect
  x = numpy.zeros((features.shape[0] + 2 * win, features.shape[1]), dtype=features.dtype)
  x[:win, :] = features[0, :]
  x[win:-win, :] = features
  x[-win:, :] = features[-1, :]

  delta = numpy.zeros(x.shape, dtype=features.dtype)

  if method == 'diff':
    filt = numpy.zeros(2 * win + 1, dtype=features.dtype)
    filt[0] = -1
    filt[-1] = 1

  for i in range(features.shape[1]):
    delta[:, i] = numpy.convolve(features[:, i], filt)

  return delta[win:-win, :]
