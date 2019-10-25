import tensorflow as tf
from gaia.utils import distance_loss, sigmoid, make_prior, shape
from tensorflow import layers

#  from tensorflow import layers
import numpy as np
tfd = tf.contrib.distributions


class AE(object):
    def __init__(self, **params):

        self.__dict__.update(params)

        self.encoder_dims = [
            [32, 3, 1],  # 64
            [32, 3, 2], # 64
            [64, 3, 1],  # 64
            [64, 3, 2], # 64
            [128, 3, 1],  # 64
            [128, 3, 2], # 64
            [128, 3, 1], # 64
            [256, 0, 0], # 8
            [256, 0, 0], # 8
        ]
        self.decoder_dims = self.encoder_dims[::-1]

        # initialize graph
        self.graph = tf.Graph()
        # initialize config
        self.config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=True)
        self.config.gpu_options.allocator_type = 'BFC'
        self.config.gpu_options.allow_growth = True
        # initialize session
        self.sess = tf.InteractiveSession(graph=self.graph, config=self.config)
        # Global step needs to be defined to coordinate multi-GPU
        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        if self.network_type == 'VAE':
            self.init_VAE()
        elif self.network_type == 'AE':
            self.init_AE()
        elif self.network_type == 'GAIA':
            self.init_GAIA()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()  # initialize network saver
        print('Network Initialized')

        


    def init_AE(self):

        self.X = tf.placeholder(tf.float32, [self.batch_size, np.prod(self.dims)])
        with tf.variable_scope("enc"):
            self.Z_G = self.encoder(self.X)
        with tf.variable_scope("dec"):
            self.X_G = self.decoder(self.Z_G)

        self.G_loss = self.reg(self.X - self.X_G)
        # specify loss to parameters
        self.params = tf.trainable_variables()
        self.G_params = [i for i in self.params if (('enc/' in i.name) or ('dec/' in i.name))]
        self.G_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.G_grads = self.G_opt.compute_gradients(self.G_loss, var_list=self.G_params)
        self.G_train = self.G_opt.apply_gradients(self.G_grads, global_step=self.global_step)

    def init_VAE(self):

        self.X = tf.placeholder(tf.float32, [self.batch_size, np.prod(self.dims)])

        # Define the model.
        self.prior = make_prior(code_size=self.n_Z)
        with tf.variable_scope("enc"):
            self.Z_G, self.posterior = self.encoder(self.X)
        self.posterior_sample = self.posterior.sample()

        # Define the loss.
        with tf.variable_scope("dec"):
            self.likelihood = self.decoder(self.posterior_sample).log_prob(self.X)
        with tf.variable_scope("dec", reuse=True):
            self.X_G = self.decoder(self.Z_G).mean()

        self.divergence = tfd.kl_divergence(self.posterior, self.prior)
        self.G_loss = - tf.reduce_mean(self.likelihood - self.divergence)

        # specify loss to parameters
        self.params = tf.trainable_variables()
        self.G_params = [i for i in self.params if (('enc/' in i.name) or ('dec/' in i.name))]
        self.G_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.G_grads = self.G_opt.compute_gradients(self.G_loss, var_list=self.G_params)
        self.G_train = self.G_opt.apply_gradients(self.G_grads, global_step=self.global_step)

    def init_GAIA(self):
        """ Initialization specific to GAIA network
        """

        self.X = tf.placeholder(tf.float32, [self.batch_size, np.prod(self.dims)])

        # run the x input through the network
        with tf.variable_scope("generator"):
            with tf.variable_scope("enc"):
                self.Z_G = self.encoder(self.X)

            # interp
            self.midpoints = tf.expand_dims(tf.random_normal(
                shape=(self.batch_size,), mean=0.5, stddev=0.25), 1)
            # flip G_Z
            Z_G_flipped = tf.concat(axis=0, values=[
                tf.slice(self.Z_G, begin=[self.batch_size//2, 0],
                         size=[self.batch_size//2, self.n_Z]),
                tf.slice(self.Z_G, begin=[0, 0],
                         size=[self.batch_size//2, self.n_Z]),

            ])
            # get midpoints
            self.Zi_G = (self.Z_G*self.midpoints) + (Z_G_flipped * (1. - self.midpoints))

            # run real images through the first autoencoder (the generator)
            with tf.variable_scope("dec"):
                self.X_G = self.decoder(self.Z_G)  # fake generated image

            # run the sampled generator_z through the decoder of the generator
            with tf.variable_scope("dec", reuse=True):
                self.Xi_G = self.decoder(self.Zi_G)  # fake generated image

        with tf.variable_scope("descriminator"):
            # Run the real x through the descriminator
            with tf.variable_scope("enc"):
                self.Z_D_X = self.encoder(self.X, D=True)  # get z from the input
                # z value in the descriminator for the real image

            with tf.variable_scope("dec"):
                self.X_D_X = self.decoder(self.Z_D_X)  # get output from z

            # run the generated x which is autoencoding the real values through the network
            with tf.variable_scope("enc", reuse=True):
                self.Z_D_G_X = self.encoder(self.X_G, D=True)  # get z from the input

            with tf.variable_scope("dec", reuse=True):
                self.X_D_G_X = self.decoder(self.Z_D_G_X)  # get output from z

            # run the interpolated (generated) x through the discriminator
            with tf.variable_scope("enc", reuse=True):
                self.Z_D_G_Zi = self.encoder(self.Xi_G, D=True)  # get z from the input

            # run gen_x through the autoencoder, return the output
            with tf.variable_scope("dec", reuse=True):
                self.X_D_G_Zi = self.decoder(self.Z_D_G_Zi)  # get output from z

        # compute losses of the model
        self.X_D_G_X_loss = self.reg(self.X - self.X_D_G_X)
        self.X_D_G_Zi_loss = self.reg(self.Xi_G - self.X_D_G_Zi)
        self.X_G_loss = (self.X_D_G_Zi_loss + self.X_D_G_X_loss)/2.

        # compute losses of the model
        self.X_D_X_loss = self.reg(self.X - self.X_D_X)

        # distance loss
        self.distance_loss = distance_loss(self.X, self.Z_G)

        # squash with a sigmoid based on the learning rate
        self.D_lr = sigmoid(self.X_D_X_loss - self.X_G_loss, shift=0., mult=self.lr_sigma_slope)
        self.G_lr = (tf.constant(1.0) - self.D_lr)*self.lr
        self.D_lr = self.D_lr*self.lr

        self.D_prop_gen = tf.clip_by_value(
            sigmoid(self.X_D_G_Zi_loss*self.sigma - self.X_D_X_loss,
                    shift=0., mult=self.lr_sigma_slope),
            0., 0.9)  # hold the discrim proportion fake aways at less than half
        self.D_prop_real = tf.constant(1.)

        # add losses for generator and descriminator
        # loss of Encoder/Decoder: reconstructing x_real well and x_fake poorly
        self.D_loss = (self.X_D_X_loss*self.D_prop_real - self.X_G_loss * self.D_prop_gen)

        # hold the discrim proportion fake aways at less than half
        self.G_prop_i = tf.clip_by_value(
            sigmoid(self.X_D_G_Zi_loss - self.X_D_G_X_loss, shift=0., mult=self.lr_sigma_slope),
            0., 1.0)

        # Generator should be balancing the reproduction
        self.G_loss = ((self.G_prop_i*self.X_D_G_Zi_loss +
                        (1.0 - self.G_prop_i) * self.X_D_G_X_loss) +
                       self.latent_loss_weights*self.distance_loss)

        # apply optimizers
        self.D_opt = tf.train.AdamOptimizer(learning_rate=self.D_lr)
        self.G_opt = tf.train.AdamOptimizer(learning_rate=self.G_lr)

        # specify loss to parameters
        self.params = tf.trainable_variables()

        self.D_params = [i for i in self.params if 'descriminator/' in i.name]
        self.G_params = [i for i in self.params if 'generator/' in i.name]

        # Calculate the gradients for the batch of data on this CIFAR tower.
        self.D_grads = self.D_opt.compute_gradients(self.D_loss, var_list=self.D_params)
        self.G_grads = self.G_opt.compute_gradients(self.G_loss, var_list=self.G_params)

        #
        self.D_train = self.D_opt.apply_gradients(self.D_grads, global_step=self.global_step)
        self.G_train = self.G_opt.apply_gradients(self.G_grads, global_step=self.global_step)

    def encoder(self, X, D=False):
        if self.layer_type == 'conv':
            return self.encoder_conv(X, D)
        else:
            return self.encoder_fc(X, D)

    def decoder(self, Z):
        if self.layer_type == 'conv':
            return self.decoder_conv(Z)
        else:
            return self.decoder_fc(Z)

    def encoder_conv(self, X, D=False, verbose=True):
        """ Draws the encoder of the network
        """
        enc_net = [tf.reshape(X, [self.batch_size, self.dims[0], self.dims[1], self.dims[2]])]
        for lay_num, (filters, kernel_size, stride) in enumerate(self.encoder_dims):
            if kernel_size > 0:  # if this is a convolutional layer
                if lay_num == len(self.encoder_dims)-1:  # if this is the last layer

                    enc_net.append(tf.contrib.layers.flatten(layers.conv2d(enc_net[len(enc_net)-1], filters=filters,
                                                                           kernel_size=kernel_size, strides=stride, padding='same',
                                                                           name='enc_'+str(lay_num),
                                                                           activation=self.default_act)))
                else:
                    if self.encoder_dims[lay_num + 1][1] == 0:
                        # flatten this layer
                        enc_net.append(tf.contrib.layers.flatten(layers.conv2d(enc_net[len(enc_net)-1], filters=filters,
                                                                               kernel_size=kernel_size, strides=stride, padding='same',
                                                                               name='enc_' +
                                                                               str(lay_num),
                                                                               activation=self.default_act)))
                    else:
                        enc_net.append(layers.conv2d(enc_net[len(enc_net)-1], filters=filters, kernel_size=kernel_size,
                                                     strides=stride, padding='same', name='enc_'+str(lay_num),
                                                     activation=self.default_act))
            else:
                enc_net.append(layers.dense(enc_net[len(enc_net)-1], units=filters, name='enc_'+str(lay_num),
                                            activation=self.default_act))
        enc_shapes = [shape(i) for i in enc_net]
        # append latent layer
        
        enc_net.append(layers.dense(
            enc_net[len(enc_net)-1], units=self.n_Z, activation=None, name='latent_layer'))  # 32, 2
        if verbose:
            print('Encoder shapes: ', enc_shapes)
        return enc_net[-1]

    def decoder_conv(self, Z, verbose=True):
        """ Draws the decoder fo the network
        """
        dec_net = [Z]
        prev_dec_shape = None
        num_div = len([stride for lay_num, (filters, kernel_size, stride)
                       in enumerate(self.decoder_dims) if stride == 2])
        cur_shape = int(self.dims[1]/(2**(num_div-1)))

        for lay_num, (filters, kernel_size, stride) in enumerate(self.decoder_dims):
            #print( [i for i in tf.trainable_variables() if 'generator/' in i.name])
            if kernel_size > 0:  # if this is a convolutional layer

                # this is the first layer and the first convolutional layer
                if (lay_num == 0) or (self.decoder_dims[lay_num - 1][1] == 0):
                    dec_net.append(tf.reshape(layers.dense(dec_net[len(dec_net)-1], cur_shape*cur_shape*filters, name='dec_'+str(lay_num),
                                                           activation=self.default_act),
                                              [self.batch_size,  cur_shape, cur_shape, filters]))
                elif stride == 2:  # if the spatial size of the previous layer is greater than the image size of the current layer
                    # we need to resize the current network dims
                    cur_shape *= 2
                    dec_net.append(tf.image.resize_nearest_neighbor(
                        dec_net[len(dec_net)-1], (cur_shape, cur_shape)))

                elif lay_num == len(self.decoder_dims)-1:  # if this is the last layer
                    # append a normal layer

                    dec_net.append((layers.conv2d(dec_net[len(dec_net)-1], filters=filters, kernel_size=kernel_size,
                                                  strides=1, padding='same', name='dec_'+str(lay_num),
                                                  activation=self.default_act)))

                # If the next layer is not convolutional but this one is
                elif self.decoder_dims[lay_num + 1][1] == 0:
                    # flatten this layers
                    dec_net.append(tf.contrib.layers.flatten(layers.conv2d(dec_net[len(dec_net)-1], filters=filters,
                                                                           kernel_size=kernel_size, strides=1, padding='same',
                                                                           name='dec_'+str(lay_num),
                                                                           activation=self.default_act)))
                else:
                    # append a normal layer
                    dec_net.append((layers.conv2d(dec_net[len(dec_net)-1], filters=filters, kernel_size=kernel_size,
                                                  strides=1, padding='same', name='dec_'+str(lay_num),
                                                  activation=self.default_act)))
            else:  # if this is a dense layer
                # append the dense layer
                dec_net.append(layers.dense(dec_net[len(dec_net)-1], units=filters, name='dec_'+str(lay_num),
                                            activation=self.default_act))

                # append the output layer
        if (self.dims[0] != shape(dec_net[-1])[1]) & (self.dims[1] != shape(dec_net[-1])[2]):
            print('warning: shape does not match image shape')
            dec_net.append(tf.image.resize_nearest_neighbor(
                dec_net[len(dec_net)-1], (self.dims[0], self.dims[1])))
        dec_net.append(layers.conv2d(
            dec_net[len(dec_net)-1], self.dims[2], 1, strides=1, activation=tf.sigmoid, name='output_layer'))
        dec_net.append(tf.contrib.layers.flatten(dec_net[len(dec_net)-1]))

        if verbose:
            print('Decoder shapes: ', [shape(i) for i in dec_net])
        return dec_net[-1]


    def encoder_fc(self, X, D=False):
        x = tf.layers.flatten(X)
        for lay in range(self.n_layers):
            x = tf.layers.dense(x, units=self.n_neurons, activation=self.default_act)
        if ((self.D_nz is not None) and (D is True)):
            loc = tf.layers.dense(x, self.D_nz)
        else:
            loc = tf.layers.dense(x, self.n_Z)
        if self.network_type != 'VAE':
            return loc
        else:
            scale = tf.layers.dense(x, units=self.n_Z, activation=tf.nn.softplus)
            return loc, tfd.MultivariateNormalDiag(loc, scale)


    def decoder_fc(self, Z):
        x = Z
        for lay in range(self.n_layers):
            x = tf.layers.dense(x, units=self.n_neurons, activation=self.default_act)
        logit = tf.layers.dense(x, units=np.prod(self.dims))
        logit = tf.reshape(logit, [-1] + [np.prod(self.dims)])
        if self.network_type != 'VAE':
            return logit
        else:
            return tfd.Independent(tfd.Bernoulli(logit), 2)

    def save_network(self, save_location, verbose=True):
        """ Save the network to some location"""
        self.saver.save(self.sess, ''.join([save_location]))
        if verbose:
            print('Network Saved')

    def load_network(self, load_location, verbose=True):
        """ Retrieve the network from some location"""
        self.saver = tf.train.import_meta_graph(load_location + '.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(
            '/'.join(load_location.split('/')[:-1]) + '/'))
        if verbose:
            print('Network Loaded')



"""

 def encoder_conv2(self, X, D=False):

        x = [tf.reshape(X, [self.batch_size]+self.dims)]


        for conv_layer in np.arange(self.n_conv_layers):
            x.append(layers.conv2d(x[-1], filters=self.n_conv_neurons*(2**conv_layer),
             kernel_size=3, strides=1,
                              padding='same',
                              activation=self.default_act))
            x.append(layers.conv2d(x[-1], filters=self.n_conv_neurons*(2**conv_layer),
             kernel_size=3, strides=2,
                              padding='same',
                              activation=self.default_act))

        x_flat = tf.contrib.layers.flatten(x[-1])

        # dense layers
        for lay in range(self.n_layers):
            x.append(tf.layers.dense(x_flat, units=self.n_neurons, activation=self.default_act))

        print('encoder shapes: ', [shape(i) for i in x])

        if ((self.D_nz is not None) and (D is True)):
            loc = tf.layers.dense(x[-1], self.D_nz)
        else:
            loc = tf.layers.dense(x[-1], self.n_Z)
        if self.network_type != 'VAE':
            return loc
        else:
            scale = tf.layers.dense(x[-1], units=self.n_Z,
                                    activation=tf.nn.softplus)
            return loc, tfd.MultivariateNormalDiag(loc, scale)


    def decoder_conv2(self, Z):
        x = [Z]
        # fully connected layers
        for lay in range(self.n_layers):
            x.append(tf.layers.dense(x[-1], units=self.n_neurons, activation=self.default_act))
        x = x[1:]
        # unflatten X
        x_conv_shape = [self.batch_size]+[self.dims[0]//(2**self.n_conv_layers),
                                            self.dims[1]//(2**self.n_conv_layers),
                                            self.n_conv_neurons*(2**self.n_conv_layers)
                                            ]
        # get layer of correct shape for convolutions
        x.append(tf.reshape(tf.layers.dense(x[-1],
            units=np.prod(x_conv_shape[1:]), activation=self.default_act), x_conv_shape))


        # convolutional layers
        for conv_layer in np.arange(self.n_conv_layers)[::-1]:
            # first resize
            x_resize = tf.image.resize_nearest_neighbor(x[-1], [shape(x[-1])[1]*2, shape(x[-1])[2]*2])
            # then perform convolution
            x.append(layers.conv2d(x_resize, filters=self.n_conv_neurons*(2**conv_layer),
                         kernel_size=3, strides=1,
                          padding='same',
                          activation=self.default_act))

            x.append(layers.conv2d(x_resize, filters=self.n_conv_neurons*(2**conv_layer),
                         kernel_size=3, strides=1,
                          padding='same',
                          activation=self.default_act))



        # final x should have the same # of channels as the input
        x.append(layers.conv2d(x[-1], filters=self.dims[2],
                     kernel_size=1, strides=1,
                      padding='same',
                      activation=tf.sigmoid))

        print('decoder shapes: ', [shape(i) for i in x])
        if list(shape(x[-1])) != [self.batch_size ]+ self.dims:
            print('Shape of output does not match input')

        x.append(tf.contrib.layers.flatten(x[-1]))
        logit = tf.layers.dense(x[-1], units=np.prod(self.dims))
        logit = tf.reshape(logit, [-1] + [np.prod(self.dims)])
        if self.network_type != 'VAE':
            return logit
        else:
            return tfd.Independent(tfd.Bernoulli(logit), 2)
"""