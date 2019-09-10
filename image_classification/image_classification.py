import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from PIL import Image


labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


batch_size = 32
nclasses = 10
nepochs = 100
weight_decay = 1.0e-4
lr = 1.0e-4


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

y_train = tf.keras.utils.to_categorical(y_train, nclasses)
y_test = tf.keras.utils.to_categorical(y_test, nclasses)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


#-----------------------------------------------------------------------------------

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(nclasses, activation='softmax'))

optim = tf.keras.optimizers.Adam(lr)

model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

model.summary()
#-----------------------------------------------------------------------------------

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, validation_split=0.1)

model.fit(x_train, y_train, batch_size=batch_size, epochs=nepochs)
model.save('mymodel.h5')

#-----------------------------------------------------------------------------------

y_pred = model.predict(x_test)

ncorrect = 0
for i in range(x_test.shape[0]):
   if (np.argmax(y_pred[i,:]) == np.argmax(y_test[i,:])):
           ncorrect += 1

print('test set accuracy (%): ', float(ncorrect)/float(x_test.shape[0])*100.0)

#-----------------------------------------------------------------------------------

