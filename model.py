import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator

# Training data loading and preprocessing
train_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
train_df.head()
train_df.shape

train_y = train_df['label'].values
train_df.drop('label', inplace=True, axis=1)
train_x = train_df.values
train_x = train_x.reshape(-1, 28, 28, 1)

# Testing data loading
test_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')
test_df.head()
test_y = test_df['label'].values
test_df.drop('label', inplace=True, axis=1)
test_x = test_df.values
test_x = test_x.reshape(-1, 28, 28, 1)

# Model Training

img_gen = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
                             width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.1, horizontal_flip=True,
                             rescale=1/255.0, validation_split=0.2)
train_gen = img_gen.flow(train_x, train_y, subset='training')
valid_gen = img_gen.flow(train_x, train_y, subset='validation')
test_gen = ImageDataGenerator(rescale=1/255.0).flow(test_x, test_y)

model = Sequential([])

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3,), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(26, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class Model_Callback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.995:
            print('Model training complete')
            self.model.stop_training = True
callback = Model_Callback()

learning_rate = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = model.fit(train_gen, epochs=50, validation_data=valid_gen, callbacks=[callback, learning_rate])
