import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras import backend as K
import librosa , librosa.display
import os 
import glob


def extract_feature(file_name):
    
    #X, sample_rate = sf.read(file_name, dtype='float32')
    X, sample_rate = librosa.load(file_name)
    if X.ndim > 1:
        X = X[:,0]
        print(X)
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        print(sub_dir)
        if(".DS_Store" != sub_dir):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                try:
                    mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
                except Exception as e:
                    print("[Error] extract feature error. %s" % (e))
                    continue
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                # labels = np.append(labels, fn.split('/')[1])
                labels = np.append(labels, label)
            print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int)



# Get features and labels
r = os.listdir("audio/Data")
r.sort()
features, labels = parse_audio_files('audio/Data', r)
#labels = one_hot_encode(labels)
np.save('feat.npy', features)
np.save('label.npy', labels)



import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Prepare the data
X = np.load('feat.npy')
y = np.load('label.npy').ravel()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=233)

# Build the Neural Network
model = Sequential()

model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Flatten())
model.add(Dense(6, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Convert label to onehot
y_train = keras.utils.to_categorical(y_train - 1, num_classes=6)
y_test = keras.utils.to_categorical(y_test - 1, num_classes=6)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

model.fit(X_train, y_train, batch_size=64, epochs=300, verbose =1)


score, acc = model.evaluate(X_test, y_test, batch_size=16)


print('Test score:', score)
print('Test accuracy:', acc)


#Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")