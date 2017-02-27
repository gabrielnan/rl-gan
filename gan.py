import os
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Reshape, BatchNormalization, UpSampling1D, Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Flatten
from keras.optimizers import SGD


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling1D(length=2))
    model.add(Convolution1D(64, 2, border_mode='same'))
    model.add(UpSampling1D(length=2))
    model.add(Convolution1D(1, 2, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model(nb_samples=2000, nb_steps=100):
    model = Sequential()
    model.add(Convolution1D(64, 2, border_mode='same', input_shape=(1, nb_samples, nb_steps)))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Convolution1D(128, 2))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def check_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train(BATCH_SIZE):
    check_dirs('models')

    # create and reload models
    discriminator = discriminator_model()
    generator = generator_model()
    if os.path.isfile('models/discriminator'):
        discriminator.load_weights('models/discriminator')
    if os.path.isfile('models.generator'):
        generator.load_weights('models/generator')

    dg = generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer='SGD')
    dg.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))

    # start iterations for training
    for epoch in tqdm(range(100)):
        print('Number of batches')
        # for _ in range()

