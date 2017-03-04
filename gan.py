import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Reshape, BatchNormalization, Convolution1D, UpSampling2D, Convolution2D, UpSampling1D
from keras.layers.core import Activation, Flatten
from keras.optimizers import SGD, Adam

# constants
conv_length = 2
final_nb_layers = 64


def generator_conv_model(nb_steps, state_dim, noise_dim):
    model = Sequential()
    model.add(Dense(input_dim=noise_dim, output_dim=50))
    model.add(Activation('tanh'))
    model.add(Dense(25 * nb_steps * state_dim))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((25, nb_steps, state_dim), input_shape=(25 * nb_steps * state_dim,)))
    model.add(Convolution2D(40, conv_length, 1, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(Convolution2D(1, conv_length, 1, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(Reshape((nb_steps, state_dim)))
    # print model.summary()

    optimizer = SGD(lr=0.005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# if __name__ == '__main__':
#     generator_conv_model(50, 5, 25)

def generator_dense_model(nb_steps, state_dim, noise_dim):
    model = Sequential()
    model.add(Dense(input_dim=noise_dim, output_dim=50))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(nb_steps * state_dim))
    model.add(Activation('sigmoid'))
    model.add(Reshape((nb_steps, state_dim)))

    opt = Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def discriminator_conv_model(nb_steps, state_dim):
    model = Sequential()
    model.add(Reshape((1, nb_steps, state_dim), input_shape=(nb_steps, state_dim)))
    model.add(Convolution2D(20, conv_length, 1, border_mode='same', input_shape=(1, nb_steps, state_dim)))
    model.add(Activation('tanh'))
    # model.add(MaxPooling1D(pool_length=conv_length))
    model.add(Convolution2D(20, conv_length, 1))
    model.add(Activation('tanh'))
    # model.add(MaxPooling1D(pool_length=conv_length))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # print model.summary()

    optimizer = SGD(lr=0.005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def discriminator_dense_model(nb_steps, state_dim):
    model = Sequential()
    model.add(Reshape((nb_steps * state_dim), input_shape=(nb_steps, state_dim)))
    model.add(Dense(input_dim=(nb_steps, state_dim), output_dim=100))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    opt = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


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
    fig = plt.figure(figsize=(10, 8))
    plt.plot(losses['d'], label='discriminative loss')
    plt.plot(losses['g'], label='generative loss')
    plt.legend()
    plt.show()
    timer = fig.canvas.new_timer(interval=2000)
    timer.add_callback(close)
    check_dirs('plots')
    fig.savefig('plots/plot.png')
    timer.start()


def close():
    plt.close()


def get_noise(batch_size, noise_dim):
    return np.random.uniform(-1, 1, size=[batch_size, noise_dim])


def pre_train_discriminator(data, generator, discriminator, batch_size=50):
    sampled_batch = data[np.random.randint(0, data.shape[0], size=batch_size)]
    noise = get_noise(batch_size)
    generated_batch = generator.predict(noise)
    in_batch = np.concatenate((sampled_batch, generated_batch))
    out_batch = [1] * batch_size + [0] * batch_size
    discriminator.fit(in_batch, out_batch, nb_epoch=1, batch_size=128)


def get_trained_generator(data, noise_dim, batch_size=50, nb_epoch=1000, plt_freq=100, save_freq=10, use_old=False,
                          conv=True):
    # define loss lists
    losses = {'d': [], 'g': []}
    nb_samples, nb_steps, state_dim = data.shape

    # create models
    generator = generator_conv_model(nb_steps, state_dim, noise_dim)
    discriminator = discriminator_conv_model(nb_steps, state_dim)

    # reload models if exist
    initial_epoch = 0
    check_dirs('models')
    gen_dir, disc_dir, epoch_dir = get_filenames(conv)
    if use_old and os.path.isfile(gen_dir) and os.path.isfile(disc_dir):
        generator.load_weights(gen_dir)
        discriminator.load_weights(disc_dir)
        with open(epoch_dir, 'r') as epoch_file:
            initial_epoch = int(epoch_file.read()) + 1

    # create GAN
    gan = gan_model(generator, discriminator)
    discriminator.trainable = True

    # start iterations for training
    for epoch in tqdm(range(initial_epoch, nb_epoch)):

        # Generate batch
        noise = get_noise(batch_size, noise_dim)
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
        noise = get_noise(batch_size, noise_dim)
        g_loss = gan.train_on_batch(noise, [1] * batch_size)
        losses['g'].append(g_loss)
        discriminator.trainable = True

        if epoch % save_freq == save_freq - 1:
            save(generator, discriminator, epoch, conv)
        if epoch % plt_freq == plt_freq - 1:
            plot_loss(losses)

    save(generator, discriminator, nb_epoch)
    return generator


def get_filenames(conv):
    file_ext = ('_conv' if conv else '')
    gen_dir = 'models/generator' + file_ext
    disc_dir = 'models/discriminator' + file_ext
    epoch_dir = 'models/last_epoch' + file_ext + '.txt'
    return gen_dir, disc_dir, epoch_dir


def save(generator, discriminator, epoch, conv):
    gen_dir, disc_dir, epoch_dir = get_filenames(conv)
    generator.save_weights(gen_dir, True)
    discriminator.save_weights(disc_dir, True)
    with open(epoch_dir, 'w') as epoch_file:
        epoch_file.write(str(epoch))


class Generator(object):
    def __init__(self, data, noise_dim):
        self.noise_dim = noise_dim
        self.generator = get_trained_generator(data, noise_dim, use_old=False)

    def generate(self, size):
        noise = get_noise(size, self.noise_dim)
        return self.generator.predict(noise)
