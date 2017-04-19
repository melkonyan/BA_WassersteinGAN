from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys, inspect
import time
import scipy.misc
utils_folder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)

import utils as utils
from six.moves import xrange
from models.GAN import GAN


class WasserstienGAN(GAN):
    def __init__(self, z_dim, crop_image_size, resized_image_size, batch_size, data_dir, clip_values=(-0.01, 0.01),
                 critic_iterations=5):
        self.clip_values = clip_values
        GAN.__init__(self, z_dim, crop_image_size, resized_image_size, batch_size, data_dir, critic_iterations)

    def _generator(self, z, dims, train_phase, activation=tf.nn.relu, scope_name="generator"):
        N = len(dims)
        image_size = self.resized_image_size // (2 ** (N - 1))
        with tf.variable_scope(scope_name) as scope:
            W_z = utils.weight_variable([self.z_dim, dims[0] * image_size * image_size], name="W_z")
            h_z = tf.matmul(z, W_z)
            h_z = tf.reshape(h_z, [-1, image_size, image_size, dims[0]])
            h_bnz = utils.batch_norm(h_z, dims[0], train_phase, scope="gen_bnz")
            h = activation(h_bnz, name='h_z')
            utils.add_activation_summary(h)

            for index in range(N - 2):
                image_size *= 2
                W = utils.weight_variable([4, 4, dims[index + 1], dims[index]], name="W_%d" % index)
                b = tf.zeros([dims[index + 1]])
                deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[index + 1]])
                h_conv_t = utils.conv2d_transpose_strided(h, W, b, output_shape=deconv_shape)
                h_bn = utils.batch_norm(h_conv_t, dims[index + 1], train_phase, scope="gen_bn%d" % index)
                h = activation(h_bn, name='h_%d' % index)
                utils.add_activation_summary(h)

            image_size *= 2
            W_pred = utils.weight_variable([4, 4, dims[-1], dims[-2]], name="W_pred")
            b = tf.zeros([dims[-1]])
            deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[-1]])
            h_conv_t = utils.conv2d_transpose_strided(h, W_pred, b, output_shape=deconv_shape)
            pred_image = tf.nn.tanh(h_conv_t, name='pred_image')
            utils.add_activation_summary(pred_image)

        return pred_image

    def _discriminator(self, input_images, dims, train_phase, activation=tf.nn.relu, scope_name="discriminator",
                       scope_reuse=False):
        N = len(dims)
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = input_images
            skip_bn = True  # First layer of discriminator skips batch norm
            for index in range(N - 2):
                W = utils.weight_variable([4, 4, dims[index], dims[index + 1]], name="W_%d" % index)
                b = tf.zeros([dims[index+1]])
                h_conv = utils.conv2d_strided(h, W, b)
                if skip_bn:
                    h_bn = h_conv
                    skip_bn = False
                else:
                    h_bn = utils.batch_norm(h_conv, dims[index + 1], train_phase, scope="disc_bn%d" % index)
                h = activation(h_bn, name="h_%d" % index)
                utils.add_activation_summary(h)

            W_pred = utils.weight_variable([4, 4, dims[-2], dims[-1]], name="W_pred")
            b = tf.zeros([dims[-1]])
            h_pred = utils.conv2d_strided(h, W_pred, b)
        return None, h_pred, None  # Return the last convolution output. None values are returned to maintatin disc from other GAN

    def _gan_loss(self, logits_real, logits_fake, feature_real, feature_fake, use_features=False):
        self.discriminator_loss = tf.reduce_mean(logits_real - logits_fake)
        self.gen_loss = tf.reduce_mean(logits_fake)

        tf.summary.scalar("Discriminator_loss", self.discriminator_loss)
        tf.summary.scalar("Generator_loss", self.gen_loss)

    def create_network(self, generator_dims, discriminator_dims, optimizer="Adam", learning_rate=2e-4,
                       optimizer_param=0.9, improved_gan_loss=True):
        super().create_network(generator_dims, discriminator_dims, optimizer, learning_rate, optimizer_param, improved_gan_loss)
        self.clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
                                     var in self.discriminator_variables]

    def dis_post_update(self):
        self.sess.run(self.clip_discriminator_var_op, feed_dict=self.get_feed_dict(True))
