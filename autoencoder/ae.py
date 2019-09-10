import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt



(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

print(x_train.shape, x_test.shape)

# flatten
x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))

print(x_train.shape, x_test.shape)


img_dim = 784
zdim = 32
hid1 = 400
hid2 = 256
lr = 1.0e-3
nepochs = 100



input_img = Input(shape=(img_dim,))
enc1 = Dense(hid1, activation='relu', activity_regularizer=regularizers.l2(1.0e-4))(input_img)
enc2 = Dense(hid2, activation='relu', activity_regularizer=regularizers.l2(1.0e-4))(enc1)
z = Dense(zdim, activation='relu')(enc2)

encoder = Model(input_img, z)


dec1 = Dense(hid2, activation='relu', activity_regularizer=regularizers.l2(1.0e-4))(z)
dec2 = Dense(hid1, activation='relu', activity_regularizer=regularizers.l2(1.0e-4))(dec1)
reconst = Dense(img_dim, activation='sigmoid')(dec2)

autoencoder = Model(input_img, reconst)

print('AE layers: ', autoencoder.layers)

input_dec = Input(shape=(zdim,))

output_dec = autoencoder.layers[-3](input_dec) 
output_dec = autoencoder.layers[-2](output_dec)
output_dec = autoencoder.layers[-1](output_dec)

decoder = Model(input_dec, output_dec)


optim = tf.keras.optimizers.Adam(learning_rate=lr)
autoencoder.compile(optimizer=optim, loss='binary_crossentropy')

print(autoencoder.summary())


# train AE
palo = autoencoder.fit(x_train, x_train, epochs=nepochs, batch_size=256, verbose=2, shuffle=True, validation_data=(x_test, x_test))
palo = palo.history
train_loss = palo['loss']
val_loss = palo['val_loss']

print('train loss: ', train_loss)
print('val loss: ', val_loss)


# test AE
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# save model
autoencoder.save('AE.h5')


decoded_imgs = decoded_imgs.reshape((-1,28,28))
print(decoded_imgs.shape)

# plot
for i in range(16):
   idx = np.random.randint(0, x_test.shape[0], 1)
   ground_truth = x_test[idx,:].reshape(28,28)
   reconstructed = decoded_imgs[idx,:].reshape(28,28)
   myfig = np.zeros((28,28*2+4),dtype=np.float32)
   myfig[:,:28] = ground_truth
   myfig[:,-28:] = reconstructed
   myfig *= 255.0
   myfig = myfig.astype(np.uint8)
   plt.imshow(myfig)
   plt.axis('off')
   plt.savefig('samples/fig_' + str(i) + '.png')
   plt.close()
