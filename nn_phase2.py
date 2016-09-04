# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 23:09:01 2016

@author: liam

1. Convert wav to a few 2 minute clips = 512 * 20 desired time dim
2. Take a 2 minute clip, divide it into the 20 sections
3. Predict each section using current NN
4. Concatenate sections
5. Predict target vector using concatenated sections

"""

from os import listdir
import numpy as np
from matplotlib import pyplot as plt
import cPickle
from NeuralSounds import *

'''
In this relative path, you need a bunch of mp3s that have the correct BPM
stored in their ID3 tag data, start on the first beat, and don't have any
skipped beats, tempo changes, etc... at least not in the first minute
'''
TRACKS_PATH = '../tracks/bpmd_wavs'

# Did multiples of 96 as my GPU has 96 cores
NUM_TRAIN = 96
NUM_VALI = 96
NUM_TEST = 96

# Randomly shuffle the mp3 files
np.random.seed(2)
all_tracks = listdir(TRACKS_PATH)
np.random.shuffle(all_tracks)

# Initialise the class with a large number of samples per file - this 
# increases speed as we get many samples from one .wav at a time
ns = NeuralSounds(downsample=32,
                  num_samples_per_file=8,
                  desired_X_time_dim=160 * 32, # about 1 minute
                  fft_sample_length=768,
                  fft_step_size=512,
                  track_fnames=all_tracks[-6:],
                  tracks_path=TRACKS_PATH)

X_train, y_train, bpms_train, fnames_train = \
    ns.get_spectogram_training_set(n_batch=NUM_TRAIN)

ns.track_fnames=all_tracks[:-6]
X_val, y_val, bpms_val, fnames_val = \
    ns.get_spectogram_training_set(n_batch=NUM_VALI)
#%%
from keras.models import load_model
model = load_model('convnet.kerasmodel')
#%%
def get_phase_2_train_input_vector(X, model):
    X_out = np.zeros((X.shape[0], X.shape[2]))
    for i in range(0, x.shape[2], 160):
        print np.round(i * 1. / x.shape[2], 2), '...'
        X_out[:, i:(i+160)] = model.predict(X[:, :, i:(i+160), :])
    return X_out
    
X_train = get_phase_2_train_input_vector(X, model)
X_val = get_phase_2_train_input_vector(X_val, model)
#%%
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop, Adadelta

output_length = y_train.shape[1]
drop_in = 0.1,
drop_hid = 0.25

dense_widths = [output_length, output_length]

early = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')

phase2_model = Sequential()

#phase2_model.add(Dropout(drop_in, input_dim=X_train.shape[1]))
phase2_model.add(Dense(dense_widths[0], input_dim=X_train.shape[1]))

for w in dense_widths[1:]:
    phase2_model.add(Dense(w))
    phase2_model.add(Activation('relu'))
    if drop_hid:
        phase2_model.add(Dropout(drop_hid))
phase2_model.add(Dense(output_length))
phase2_model.add(Activation('relu'))

phase2_model.summary()

opt = Adadelta()
#opt = SGD(lr=0.01)
#opt = Adam()

phase2_model.compile(loss='mse',
              optimizer=opt,
              metrics=[])

batch_size = 96
nb_epoch = 100
history = phase2_model.fit(X_train, y_train,
                           batch_size=batch_size, nb_epoch=nb_epoch,
                           verbose=1, validation_data=(X_val, y_val),
shuffle=True, callbacks=[early])
#%%
def plot_pulses(idx, X, y, model, rng=[0,500]):
    i = model.predict(X[idx:(idx+1)])
    a = i[0, rng[0]:rng[1]]
    b = y[idx, rng[0]:rng[1]]
    plt.plot(np.vstack((a, b * 10000 - 9990)).T)
    plt.ylim([0, 1.])
    plt.xlim([0, a.shape[0]])
    plt.xlabel('Time')
    plt.ylabel('Pulse')
    plt.show()

plot_pulses(5, X_val, y_val, phase2_model)
#%%
for i in range(0, len(X_train), len(X_train)/20):
    plot_pulses(i, X_train, y_train, phase2_model)
    plt.show()
for i in range(0, len(X_train), len(X_train)/20):
    plt.plot(X_train[i][3000:3500])
    plt.show()
#%%
for i in range(0, len(X_val), len(X_val)/20):
    plot_pulses(i, X_val, y_val, phase2_model)
    plt.show()
    #%%
rng=[3000,3500]
for i in range(0, len(X_val), len(X_val)/20):
    plt.plot(X_val[i][rng[0]:rng[1]])
    plt.plot(y[i, rng[0]:rng[1]])
    plt.show()


