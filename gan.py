import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Reshape, BatchNormalization, Convolution1D
from keras.layers.core import Activation, Flatten
from keras.optimizers import SGD

# constants
first_nb_layers = 1024
conv_length = 2
final_nb_layers = 64


def generator_conv_model(nb_steps, state_dim, noise_dim):
    model = Sequential()
    model.add(Dense(input_dim=noise_dim, output_dim=first_nb_layers))
    model.add(Activation('tanh'))
    model.add(Dense(nb_steps * state_dim))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((nb_steps, state_dim), input_shape=(nb_steps * state_dim,)))
    model.add(Convolution1D(100, conv_length, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(Convolution1D(1, conv_length, border_mode='same'))
    model.add(Activation('tanh'))
    optimizer = SGD(lr=0.005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def generator_dense_model(nb_steps, state_dim):
    pass


def discriminator_conv_model(nb_steps, state_dim):
    model = Sequential()
    model.add(Convolution1D(final_nb_layers, conv_length, border_mode='same', input_shape=(1, nb_steps, state_dim)))
    model.add(Activation('tanh'))
    # model.add(MaxPooling1D(pool_length=conv_length))
    model.add(Convolution1D(100, conv_length))
    model.add(Activation('tanh'))
    # model.add(MaxPooling1D(pool_length=conv_length))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    optimizer = SGD(lr=0.005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def discriminator_dense_model(nb_steps, state_dim):
    pass


def gan_model(generator, discriminator):
    gan = Sequential()
    gan.add(generator)
    discriminator.trainable = False
    gan.add(discriminator)
    gan.compile(loss='binary_crossentropy', optimizer='SGD')
    return gan


def check_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_loss(losses):
    plt.figure(figsize=(10, 8))
    plt.plot(losses['d'], label='discriminative loss')
    plt.plot(losses['g'], label='generative loss')
    plt.legend()
    plt.show()


def get_noise(batch_size, noise_dim=100):
    return np.random.uniform(-1, 1, size=[batch_size, noise_dim])


def pre_train_discriminator(data, generator, discriminator, batch_size=50):
    sampled_batch = data[np.random.randint(0, data.shape[0], size=batch_size)]
    noise = get_noise(batch_size)
    generated_batch = generator.predict(noise)
    in_batch = np.concatenate((sampled_batch, generated_batch))
    out_batch = [1] * batch_size + [0] * batch_size
    discriminator.fit(in_batch, out_batch, nb_epoch=1, batch_size=128)


def get_trained_generator(data, batch_size=50, nb_epoch=100, plt_freq=25, save_freq=10, noise_dim=100):

    # define loss lists
    losses = {'d': [], 'g': []}
    nb_samples, nb_steps, state_dim = data.shape

    # create models
    generator = generator_conv_model(nb_steps, state_dim, noise_dim)
    discriminator = discriminator_conv_model(nb_steps, state_dim)

    # reload models if exist
    initial_epoch = 0
    check_dirs('models')
    if os.path.isfile('models/discriminator') and os.path.isfile('models/generator'):
        discriminator.load_weights('models/discriminator')
        generator.load_weights('models/generator')
        with open('models/last_epoch.txt', 'r') as epoch_file:
            initial_epoch = int(epoch_file.read()) + 1

    # create GAN
    gan = gan_model(generator, discriminator)
    discriminator.trainable = True

    # start iterations for training
    for epoch in tqdm(range(initial_epoch, nb_epoch)):

        # Generate batch
        noise = get_noise(batch_size)
        generated_batch = generator.predict(noise, verbose=0)

        # Sample batch
        sampled_batch = data[np.random.randint(0, data.shape[0], size=batch_size)]

        # Concat generated and sampled batches
        in_batch = np.concatenate((sampled_batch, generated_batch))
        out_batch = [1] * batch_size + [0] * batch_size

        # Train discriminator
        d_loss = discriminator.train_on_batch(in_batch, out_batch)
        losses['d'].append(d_loss)

        # Train generator inside the GAN
        discriminator.trainable = False
        noise = get_noise(batch_size)
        g_loss = gan.train_on_batch(noise, [1] * batch_size)
        losses['g'].append(g_loss)
        discriminator.trainable = True

        if epoch % save_freq == save_freq - 1:
            save(generator, discriminator, epoch)
        if epoch % plt_freq == plt_freq - 1:
            plot_loss(losses)

    save(generator, discriminator, nb_epoch)
    return generator


def save(generator, discriminator, epoch):
    generator.save_weights('models/generator', True)
    discriminator.save_weights('models/discriminator', True)
    with open('models/last_epoch.txt', 'w') as epoch_file:
        epoch_file.write(epoch)

