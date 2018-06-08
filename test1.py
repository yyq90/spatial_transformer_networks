
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from scipy.misc import imresize
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD

import keras.backend as K
from src.spatial_transformer import SpatialTransformer
import dataLoad

nb_epoch = 12

batch_size = 128
nb_classes = 100
data_augmentation = True

# input image dimensions
# img_rows, img_cols = 32, 32
img_rows, img_cols = 256, 256
# The CIFAR10 images are RGB.
# img_channels = 3
img_channels = 3

# The data, shuffled and split between train and test sets:
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train,y_train,X_test,y_test= dataLoad.dataload(img_rows,img_cols,gray=0)

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.








# reshape for convolutions




input_shape =  np.squeeze(X_train.shape[1:])
input_shape = (img_rows,img_cols,img_channels)
print("Input shape:",input_shape)




# plt.figure(figsize=(7,7))
# plt.imshow(X_train[132],  interpolation='none')
# plt.title('Cluttered', fontsize=20)
# plt.axis('off')
# plt.show()



# initial weights
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]


locnet = Sequential()
locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape,name='pooling1_input'))
locnet.add(Convolution2D(20, (5, 5),name='conv1'))
locnet.add(MaxPooling2D(pool_size=(2,2),name='pooling2'))
locnet.add(Convolution2D(20, (5, 5),name='conv2'))

locnet.add(Flatten())
locnet.add(Dense(50,name='dense1'))
locnet.add(Activation('relu',name='Activation1'))
locnet.add(Dense(6, weights=weights))
#locnet.add(Activation('sigmoid'))


model = Sequential()

model.add(SpatialTransformer(localization_net=locnet,
                             output_size=(30,30), input_shape=input_shape))

model.add(Convolution2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print(locnet.summary())
print(model.summary())

XX = model.input
YY = model.layers[0].output
F = K.function([XX], [YY])

print(X_train.shape[0]/batch_size)

nb_epochs = 10  # you probably want to go longer than this
fig = plt.figure()
try:
    for e in range(nb_epochs):
        print('-' * 40)
        # progbar = generic_utils.Progbar(X_train.shape[0])
        for b in range(150):
            # print(b)
            f = b * batch_size
            l = (b + 1) * batch_size
            X_batch = X_train[f:l].astype('float32')
            Y_batch = Y_train[f:l].astype('float32')
            loss = model.train_on_batch(X_batch, Y_batch)
            # print(loss)
            # progbar.add(X_batch.shape[0], values=[("train loss", loss)])
        # scorev = model.evaluate(X_valid, y_valid, verbose=1)
        scoret = model.evaluate(X_test, Y_test, verbose=1)
        # print('Epoch: {0} | Valid: {1} | Test: {2}'.format(e, scorev, scoret))
        print('Epoch: {0} | Valid: {1} | Test: {2}'.format(e, 0, scoret))

        if e % 1 == 0:
            Xresult = F([X_batch[:9]])
            plt.clf()
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                image = np.squeeze(Xresult[0][i])
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            fig.canvas.draw()
            plt.show()

except KeyboardInterrupt:
    pass









Xaug = X_train[:9]
Xresult = F([Xaug.astype('float32')])




# input
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(np.squeeze(Xaug[i]), cmap='gray')
    plt.axis('off')



# output
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(np.squeeze(Xresult[0][i]), cmap='gray')
    plt.axis('off')