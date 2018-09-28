# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

ASR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ASR_ROOT)

from ASR.src.util.audio import extractMFCC
from ASR.src.base import fparam

wav_dir = '/home/guwenqi/Documents/jdd_digits/wav'
feature_dir = '/home/guwenqi/Documents/jdd_digits/feature'

i = 0
for root, dirlist, filelist in os.walk(wav_dir):
  try:
    os.makedirs(root.replace(wav_dir, feature_dir))
  except:
    pass
  for filename in filelist:
    if filename.endswith('.wav'):
      fea = extractMFCC(os.path.join(root, filename),
                        fparam.MFCC_ORDER,
                        frame_length=fparam.MFCC_FRAME_LENGTH,
                        frame_shift=fparam.MFCC_FRAME_SHIFT)
      np.save(os.path.join(root, filename).replace(wav_dir, feature_dir).replace('.wav', ''), fea)
      i += 1
      if i% 100 == 0:
        print (i)
