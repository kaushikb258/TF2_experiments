import numpy as np
import sys
import tensorflow as tf

from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import ModelCheckpoint

tf.compat.v1.estimator.Estimator

print('GPU used: ', tf.test.is_gpu_available())


batch_size = 128
nepochs = 100
nclasses = 10
lr = 1.0e-3
width = 28
height = 28


fashion_labels = ["Shirt/top", "Trousers", "Pullover", "Dress", "Coat", "sandal", "shirt", "Sneaker", "Bag", "Ankle boot"]

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('train: ', x_train.shape, y_train.shape)
print('test: ', x_test.shape, y_test.shape)

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0


x_train = x_train.reshape((x_train.shape[0], width*height))
x_test = x_test.reshape((x_test.shape[0], width*height))

split = 50000

xtrain = x_train[:split]
xval = x_train[split:]
ytrain = y_train[:split]
yval = y_train[split:]


ytrain_1hot = tf.one_hot(ytrain, depth=nclasses).numpy()
yval_1hot = tf.one_hot(yval, depth=nclasses).numpy()
ytest_1hot = tf.one_hot(y_test, depth=nclasses).numpy()




inputs = tf.keras.Input(shape=(width*height))
logits = tf.keras.layers.Dense(nclasses)(inputs)
outputs = tf.nn.softmax(logits)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

checkpointer = ModelCheckpoint(filepath='./model.weights.best.hdf5', verbose=2, save_best_only=True, save_weights_only=True)
optim = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain_1hot, batch_size=batch_size, epochs=nepochs, validation_data=(xval, yval_1hot), callbacks=[checkpointer], verbose=2)

print('finished training ')

model.load_weights('model.weights.best.hdf5')

scores = model.evaluate(x_test, ytest_1hot, batch_size, verbose=2)
print('scores: ', scores)

y_preds = model.predict(x_test)



# test

index = 258

index_predicted = np.argmax(y_preds[index])
index_ground_truth = np.argmax(ytest_1hot[index])

print('index (pred/ground truth): ', index_predicted, fashion_labels[index_predicted], fashion_labels[index_ground_truth])

