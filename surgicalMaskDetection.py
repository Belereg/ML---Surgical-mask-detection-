# importing all the mandatory libraries to work with
import pandas as pd
import numpy as np
import librosa.display
import librosa
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score, f1_score
from python_speech_features import mfcc
from librosa import display
import IPython.display as ipd
from glob import glob
from matplotlib import cm
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


# defined a normalization function (lab5) with multiple preprocessing method choices
def normalize(train_features, test_features, type=None):
    if type == 'min_max':
        scaler = preprocessing.MinMaxScaler()
    elif type == 'standard':
        scaler = preprocessing.StandardScaler()
    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')
    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')
    elif type == 'maxabs':
        scaler = preprocessing.MaxAbsScaler()
    elif type == 'robust':
        scaler = preprocessing.RobustScaler()
    else:
        print("Error: the type is not recognized.")
        exit()

    scaler.fit(train_features)
    scaled_train_feats = scaler.transform(train_features)
    scaled_test_feats = scaler.transform(test_features)

    return scaled_train_feats, scaled_test_feats


train_mfccs_list = []
train_labels_list = []

test_mfccs_list = []

# reading and stocking the information (the name of the audio files and the label 0 or 1) from the train files,
# using panda's library method read_csv
train_files = pd.read_csv(
    "C:/Users/nalex/Desktop/competitieMLkaggle/train.txt",
    names=['audio_filename', 'label']
)

# reading and stocking the information (the name of the audio files) from the files that will be tested
test_audio_filenames = pd.read_csv(
    "C:/Users/nalex/Desktop/competitieMLkaggle/test.txt",
    names=['audio_filename']
)

for file in range(0, len(train_files.values), 1):
    # for each filename in the train.txt, i am searching for it in the actual audio files from the folder "train"
    # since they are not in the same order as in the train.txt file.
    audiofile_name = "C:/Users/nalex/Desktop/competitieMLkaggle/train\\" + train_files.values[file][0]

    # i am loading and decoding the audio as a time series.
    # "audio", is simply a numpy ndarray of floating points
    # "sr" comes from "sampling rate" and it is basically the number of samples of audio per second, and it is
    # by default 22050 Hz
    audio, sr = librosa.load(audiofile_name, res_type='kaiser_best')

    # getting the Mel-frequency cepstral coefficient (MFCC) features from each of the audio file
    # and using np.mean() function on vertical axis
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=500)
    mfccsscaled = np.mean(mfccs.T, axis=0)

    # adding the scaled mfcc to the list
    train_mfccs_list.append(mfccsscaled)

    # rendering the MFCC spectogram! (down below)
    # librosa.display.specshow(mfccs, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    # plt.title('mfcc')
    # plt.show()

# converting the simple python list of mfccs to a numpy array
train_mfccs = np.array(train_mfccs_list)

# getting only the labels (0 and 1) from the train files
for audio, label in train_files.values:
    train_labels_list.append(label)
train_labels = np.array(train_labels_list)

for file in range(0, len(test_audio_filenames.values), 1):
    # for each filename in the test.txt, i am searching for it in the actual audio files from the folder "test"
    # since they are not in the same order as in the test.txt file.
    audiofile_name = "C:/Users/nalex/Desktop/competitieMLkaggle/test\\" + test_audio_filenames.values[file][0]

    # i am loading and decoding the audio as a time series.
    # "audio", is simply a numpy ndarray of floating points
    # "sr" comes from "sampling rate" and it is basically the number of samples of audio per second, and it is
    # by default 22050 Hz
    audio, sr = librosa.load(audiofile_name, res_type='kaiser_best')

    # getting the Mel-frequency cepstral coefficient (MFCC) features from each of the audio file
    # and using np.mean() function on vertical axis
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=500)
    mfccsscaled = np.mean(mfccs.T, axis=0)

    # adding the scaled mfcc to the list
    test_mfccs_list.append(mfccsscaled)

# converting the simple python list of mfccs to a numpy array
test_mfccs = np.array(test_mfccs_list)

# using the normalize function from lab5
scaled_train, scaled_test = normalize(train_mfccs, test_mfccs, 'standard')

# training the data with svm.SVC (support vector classification) with C=1 and linear kernel
svm = svm.SVC(C=1, kernel='linear')
svm.fit(scaled_train, train_labels)

# getting the predictions with svm.predict
preds = svm.predict(scaled_test)

# opening incercare.txt file for writing the resulted labels of the test files and formatting it for kaggle submission
file = open("incercare.txt", "w+")
file.write("name,label\n")
for i in range(len(test_audio_filenames.values)):
    name = str(test_audio_filenames.values[i])
    name = name[2:-2]
    file.write(name + "," + str(preds[i]))
    file.write("\n")

print("\nProgram has ended")
