import numpy as np
import random
import math
import time

#load data
data = np.load('octave2_x_T.npy')
prev_data = np.load('octave2_prev_x_T.npy')
print('data shape: {}'.format(data.shape))
time.sleep(3)

song_idx = int(data.shape[0]/8)
test_ratial = 0.1
test_song_num = round(song_idx*test_ratial)
train_song_num = data.shape[0] - test_song_num
print('total song number: {}'.format(song_idx))
print('number of test song: {}, \n,number of train song: {}'.format(test_song_num,train_song_num))
time.sleep(3)

#create the song idx for test data

full = np.arange(song_idx)

test_idx= random.sample(range(0,full.shape[0]),test_song_num)
test_idx = np.asarray(test_idx)
print('total {} song idx for test: {}'.format(test_idx.shape[0],test_idx))
time.sleep(3)

#create the song idx for train data
train_idx = np.delete(full,test_idx)
print('total {} song idx for train: {}'.format(train_idx.shape[0],train_idx))
time.sleep(3)

    

def test_data(data,test_idx):

    #save the test data and train data separately
    X_te = []
    for i in range(test_idx.shape[0]):
        stp = (test_idx[i])*8
        edp = stp + 8
        song = data[stp:edp,0,:,:]
        song = song.reshape((8,1,128,16))
        X_te.append(song)
        # print('i: {}, test_iex: {}, stp: {}, song.shape: {}, song num: {}'.format(i, test_idx[i], stp, song.shape, len(X_te)))
        
    X_te = np.vstack(X_te)
    return X_te


def train_data(data,train_idx):

    #save the test data and train data separately
    X_tr = []
    for i in range(train_idx.shape[0]):
        stp = (train_idx[i])*8
        edp = stp + 8
        song = data[stp:edp,0,:,:]
        song = song.reshape((8,1,128,16))
        X_tr.append(song)
        # print('i: {}, train_iex: {}, stp: {}, song.shape: {}, song num: {}'.format(i, train_idx[i], stp, song.shape, len(X_tr)))

    X_tr = np.vstack(X_tr)
    return X_tr



# test_data
X_te = test_data(data,test_idx)
prev_X_te = test_data(prev_data,test_idx)
np.save('octave2_X_te.npy',X_te)
np.save('octave2_prev_X_te',prev_X_te)

print('test song completed, X_te matrix shape: {}'.format(X_te.shape))

#train_data
X_tr = train_data(data,train_idx)
prev_X_tr = train_data(prev_data,train_idx)
np.save('octave2_X_tr.npy',X_tr)
np.save('octave2_prev_X_tr.npy',prev_X_tr)

print('train song completed, X_tr matrix shape: {}'.format(X_tr.shape))






