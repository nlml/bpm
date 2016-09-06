# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:22:52 2016

Help from:
http://stackoverflow.com/a/23378284/6167850

@author: liam
"""
from os import listdir
import numpy as np
from matplotlib import pyplot as plt
import cPickle
from NeuralSounds import *

# Audio files are stored in this relative path
TRACKS_PATH = '../tracks/bpmd_wavs'

# Did multiples of 96 as my GPU has 96 cores
NUM_TRAIN = 96 * 300
NUM_VALI = 96 * 20
NUM_TEST = 96 * 50

# Randomly shuffle the mp3 files
np.random.seed(2)
all_tracks = listdir(TRACKS_PATH)
np.random.shuffle(all_tracks)

# Initialise the class with a large number of samples per file - this 
# increases speed as we get many samples from one .wav at a time
ns = NeuralSounds(downsample=32,
                  num_samples_per_file=960,
                  desired_X_time_dim=160,
                  fft_sample_length=768,
                  fft_step_size=512,
                  track_fnames=all_tracks[0:-6],
                  tracks_path=TRACKS_PATH)

X, y, bpms, fnames = ns.get_spectogram_training_set(n_batch=NUM_TRAIN)
cPickle.dump((X, y, bpms, fnames), open('Xy_pulse3.dump', 'wb'))

# Change the number of samples per file for creating the test and validation
# sets, as these are smaller so speed is not such an issue, and we want 
# variation. Also use different tracks to test/validate.
ns.num_samples_per_file = 100
ns.track_fnames = all_tracks[-6:]

X, y, bpms, fnames = ns.get_spectogram_training_set(NUM_VALI)
cPickle.dump((X, y, bpms, fnames), open('Xy_vali_pulse3.dump', 'wb'))

X, y, bpms, fnames = ns.get_spectogram_training_set(NUM_TEST)
cPickle.dump((X, y, bpms, fnames), open('Xy_test_pulse3.dump', 'wb'))

# We can plot the beat spikes (training output) over the spectograms
# (training input). The spikes should match up with the beats.
for i in xrange(0, len(y), max(len(y) / 10, 1)):
    plt.figure(figsize=(10,8))
    plt.plot(y[i] * 1000 - 900, 'black', linewidth=1)
    plt.imshow(X[i][0].T, aspect='auto', origin='top')
    plt.show()
