import tensorflow as tf
import numpy as np
import sys


print('tf version: ', tf.__version__)
print('keras version: ', tf.keras.__version__)
print('gpu used: ', tf.test.is_gpu_available())


mnist = tf.keras.datasets.mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x/255.0
test_x = test_x/255.0

print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


nepochs = 10
batch_size = 64
lr = 1.0e-3

#--------------------------------

inputs = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(512, activation='relu', name='hid1')(x)
x = tf.keras.layers.Dropout(0.2)(x)
pred = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='out')(x)

model = tf.keras.Model(inputs=inputs, outputs=pred)

print(model.summary())

optim = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs)

print('finished training ')

model.save('mnist_nn.h5')

print('eavl on test: ', model.evaluate(test_x, test_y))
