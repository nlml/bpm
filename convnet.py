'''
Neural Network for detecting the BPM of a 4 second clip of music
'''

from matplotlib import pyplot as plt
import numpy as np
import cPickle
from vec_to_bpm import vec_to_bpm
np.random.seed(1)  # for reproducibility

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
#%%
print("Loading data...")

X_train, y_train, bpms_train, fnames_train = cPickle.load(open('Xy_pulse3.dump', 'rb'))
X_val, y_val, bpms_val, fnames_val = cPickle.load(open('Xy_vali_pulse3.dump', 'rb'))
#%%
input_time_dim = X_train.shape[2]
input_freq_dim = X_train.shape[3]
output_length = y_train.shape[1]

drop_hid = 0.25
num_filters = 32
dense_widths = [output_length*2, output_length]

early = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

model = Sequential()

model.add(Convolution2D(num_filters, 3, 3, border_mode='same', 
                        input_shape=(1, input_time_dim, input_freq_dim)))
model.add(Activation('relu'))

model.add(Convolution2D(num_filters, 5, 5, border_mode='same'))
model.add(Activation('relu'))

model.add(Reshape((input_time_dim, input_freq_dim * num_filters)))

model.add(TimeDistributed(Dense(256)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(128)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(8)))
model.add(Activation('relu'))

model.add(Flatten())
if drop_hid:
    model.add(Dropout(drop_hid))

for w in dense_widths:
    model.add(Dense(w))
    model.add(Activation('relu'))
    if drop_hid:
        model.add(Dropout(drop_hid))
model.add(Dense(output_length))
model.add(Activation('relu'))

model.summary()

#opt = Adadelta()
#opt = SGD(lr=0.001)
opt = Adam()

model.compile(loss='mse',
              optimizer=opt,
              metrics=[])
#%%t
batch_size = 96 * 4
nb_epoch = 100
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, y_val),
                    shuffle=True, callbacks=[early])
#%%
model.save('convnet_val_loss=' +\
           str(np.round(history.history['val_loss'][-1], 4)) + '.kerasmodel')
#%%
def plot_pulses(idx, X, y, model):
    i = model.predict(X[idx:(idx+1)])
    plt.plot(np.vstack((i, y[idx] * 10000 - 9990)).T)
    plt.ylim([0, 1.])
    plt.xlim([0, i.shape[1]])
    plt.xlabel('Time')
    plt.ylabel('Pulse')
    plt.show()
    
# Plot actual vs predicted pulses in training set
for idx in xrange(0, len(X_train), len(X_train) / 10):
    print "train set actual / predicted pulse", idx
    plot_pulses(idx, X_train, y_train)
    
for idx in xrange(0, len(X_val), len(X_val) / 10):
    print "val set actual / predicted pulse", idx
    plot_pulses(idx, X_val, y_val)
#%%
def plot_spect_and_beats(X, y, idx, model):
    plt.figure(figsize=(12,8))
    print idx
    i = model.predict(X[idx:(idx+1)])
    # Plot actual pulse vector (white lines)
    plt.plot(y[idx] * 1000 - 900, 'white', linewidth=2)
    # Plot predicted pulse vector (black)
    plt.plot(32. * i.T, 'black', linewidth=2)
    # Plot spectogram
    plt.imshow(X_train[idx][0].T, aspect='auto', origin='top')
    plt.xlabel('Time')
    plt.ylabel('Frequency Bin')
    plt.show()
    
l = len(X_train)
s = 0
for idx in range(s, s + l, max(1, l/10)):
    plot_spect_and_beats(X_train, y_train, idx, model)
    #%%
l = len(X_val)
s = 0
for idx in range(s, s + l, max(1, l/10)):
    plot_spect_and_beats(X_val, y_val, idx, model)
    
#%%
#==============================================================================
# X_test, y_test, bpms_test, fnames_test = cPickle.load(open('Xy_test_pulse2.dump', 'rb'))
# 
# for idx, i in enumerate(X_train[0::len(X_train)/10]):
#     plt.figure(figsize=(12,8))
#     i = model.predict(i.reshape(1, *i.shape))
#     plt.plot(32. * i.T, 'black', linewidth=2)
#     #plt.plot(60. * y_train[idx], 'black', linewidth=2)
#     plt.imshow(X_train[idx][0].T, aspect='auto', origin='top')
#     plt.xlabel('Time')
#     plt.ylabel('Frequency Bin')
#     plt.show()
# #%%
# for idx, i in enumerate(model.predict(X_train[0::len(X_train)/10])):
#     print("Training set actual / predicted pulse", idx)
#     plt.figure(figsize=(12,10))
#     #plt.plot(32. * np.vstack((y_train[idx])).T, linewidth=2)
#     plt.plot(32. * np.vstack((y_train[idx], y_train[idx])).T, linewidth=2)
#     plt.imshow(X_train[idx][0].T, aspect='auto', origin='top')
#     plt.xlabel('Time')
#     plt.ylabel('Frequency Bin')
#     plt.show()
# #%%
# for idx, i in enumerate(model.predict(X_test[0::len(X_test)/10])):
#     print("Training set actual / predicted pulse", idx)
#     plt.figure(figsize=(12,10))
#     plt.plot(32. * np.vstack((i, y_test[idx])).T, linewidth=2)
#     plt.imshow(X_test[idx][0].T, aspect='auto', origin='top')
#     plt.xlabel('Time')
#     plt.ylabel('Frequency Bin')
#     plt.show()
# #%%
# errs = []
# for idx, i in enumerate(model.predict(X_val[0:100])):
#     errs.append([vec_to_bpm(i), bpms_val[idx]])
# errs = np.array(errs)
# #np.mean(np.abs(errs))
# print errs
#==============================================================================


