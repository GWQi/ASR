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

def extractMFCC(filepath, numcep, frame_length=0.025, frame_shift=0.01, n_stride=1, n_context=0):
  """
  frame_length, frame_shift in seconds.
  """
  fs, data = wavfile.read(filepath)

  # transform steoro to mono
  if data.ndim == 2:
    data = data.mean(axis=-1)

  features = python_speech_features.mfcc(data, fs, frame_length, frame_shift, numcep, nfilt=64, nfft=int(math.ceil(frame_length*fs)))

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


