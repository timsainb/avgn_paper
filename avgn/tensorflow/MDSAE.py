import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np


class MDSAE(tf.keras.Model):
    """a basic autoencoder class for tensorflow
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(MDSAE, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    @tf.function
    def encode(self, x):
        return self.enc(x)

    @tf.function
    def decode(self, z):
        return self.dec(z)

    @tf.function
    def compute_loss(self, x):
        z = self.encode(x)
        _x = self.decode(z)
        ae_loss = tf.reduce_mean(tf.square(x - _x))
        dist_loss = distance_loss(x, z)
        loss = ae_loss + dist_loss
        return loss, ae_loss, dist_loss

    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss, _, _ = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    def train_net(self, train_x):
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


def plot_reconstruction(model, example_data, nex=5, zm=3):

    example_data_reconstructed = model.decode(model.encode(example_data))
    fig, axs = plt.subplots(ncols=nex, nrows=2, figsize=(zm * nex, zm * 2))
    for exi in range(nex):
        axs[0, exi].matshow(
            example_data.numpy()[exi].squeeze(), cmap=plt.cm.Greys  # , vmin=0, vmax=1
        )
        axs[1, exi].matshow(
            example_data_reconstructed.numpy()[exi].squeeze(),
            cmap=plt.cm.Greys,
            # vmin=0,
            # vmax=1,
        )
    for ax in axs.flatten():
        ax.axis("off")
    plt.show()


def distance_loss(x1, x2):
    """ Loss based on the distance between elements in a batch
    """
    sdx = squared_dist(tf.reshape(x1, [len(x1), tf.math.reduce_prod(tf.shape(x1)[1:])]))
    sdx = sdx / tf.reduce_mean(sdx)
    sdz = squared_dist(tf.reshape(x2, [len(x2), tf.math.reduce_prod(tf.shape(x2)[1:])]))
    sdz = sdz / tf.reduce_mean(sdz)
    return tf.reduce_mean(
        tf.square(
            tf.math.log(tf.constant(1.0) + sdx) - (tf.math.log(tf.constant(1.0) + sdz))
        )
    )


def squared_dist(A):
    """
    Computes the pairwise distance between points
    #http://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    """
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_mean(tf.math.squared_difference(expanded_a, expanded_b), 2)
    return distances
