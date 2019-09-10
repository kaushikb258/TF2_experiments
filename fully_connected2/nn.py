import tensorflow as tf
import numpy as np
import sys
from sklearn.datasets import load_iris


print('tf version: ', tf.__version__)
print('keras version: ', tf.keras.__version__)
print('gpu used: ', tf.test.is_gpu_available())


iris = load_iris()

x = iris.data
y = iris.target


xmean = np.mean(x,0)
xstd = np.std(x,0)

x = (x - xmean)/xstd


x_train = []
y_train = []
x_test = []
y_test = []


for i in range(x.shape[0]):
    if (np.random.uniform() <= 0.9):
         x_train.append(x[i])
         y_train.append(y[i])
    else:
         x_test.append(x[i])
         y_test.append(y[i])

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)    

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


nepochs = 1000
lr = 1.0e-3
nclasses = 3

#--------------------------------

inputs = tf.keras.Input(shape=(4))
x = tf.keras.layers.Dense(256, activation='relu', name='hid1')(inputs)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(128, activation='relu', name='hid2')(x)
x = tf.keras.layers.Dropout(0.4)(x)
pred = tf.keras.layers.Dense(nclasses, activation=tf.nn.softmax, name='out')(x)

model = tf.keras.Model(inputs=inputs, outputs=pred)

print(model.summary())

optim = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=x_train.shape[0], epochs=nepochs)

print('finished training ')

model.save('mnist_nn.h5')

print('eavl on test: ', model.evaluate(x_test, y_test))

