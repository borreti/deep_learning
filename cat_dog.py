# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 17:39:05 2019

@author: ROBERT
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

#layers sequence: convolution, pooling e flattening

classificator = Sequential()
#conv2D(nFiltros, (tuple_kernel_size), (width, height, n_filters), activation function)
classificator.add(Conv2D(32, (3,3), input_shape =(64,64,3), activation='relu'))
classificator.add(BatchNormalization()) #batch normalization to speed up the learning
classificator.add(MaxPooling2D(pool_size=(2,2)))

#adding more layers
classificator.add(Conv2D(32, (3,3), activation='relu'))
classificator.add(BatchNormalization())
classificator.add(MaxPooling2D(pool_size=(2,2)))

#adding flatten layer
classificator.add(Flatten())

classificator.add(Dense(units = 128, activation = 'relu'))
classificator.add(Dropout(0.2))
classificator.add(Dense(units = 1, activation = 'sigmoid'))

classificator.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#augmetattion
g_training = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

g_test = ImageDataGenerator(rescale = 1./255)

base_training = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')

base_test = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')

classificator.fit_generator(base_training, steps_per_epoch = 4000 / 32,
                            epochs = 10, validation_data = base_test,
                            validation_steps = 1000 / 32)


#classify new image

imagem_teste = image.load_img('dataset/test_set/gato/cat.3500.jpg')

imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /=255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)

classificator.predict(imagem_teste)
