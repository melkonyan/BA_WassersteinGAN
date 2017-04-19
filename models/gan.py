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
import Dataset_Reader.read_celebADataset as celebA
from six.moves import xrange


class GAN(object):
    def __init__(self, z_dim, crop_image_size, resized_image_size, batch_size, data_dir, critic_iterations=5, root_scope_name=''):
        celebA_dataset = celebA.read_dataset(data_dir)
        self.root_scope_name = root_scope_name
        self.z_dim = z_dim
        self.crop_image_size = crop_image_size
        self.resized_image_size = resized_image_size
        self.batch_size = batch_size
        self.critic_iterations=critic_iterations
        filename_queue = tf.train.string_input_producer(celebA_dataset.train_images)
        self.training_batch_images = self._read_input_queue(filename_queue)

    def _read_input(self, filename_queue):
        """
        :return: a tensorflow node that will read one image from a file at a time and convert it to a tensor of floats.
        """
        class DataRecord(object):
            pass

        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        record = DataRecord()
        decoded_image = tf.image.decode_jpeg(value,
                                             channels=3)  # Assumption:Color images are read and are to be generated

        # decoded_image_4d = tf.expand_dims(decoded_image, 0)
        # resized_image = tf.image.resize_bilinear(decoded_image_4d, [self.target_image_size, self.target_image_size])
        # record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])

        cropped_image = tf.cast(
            tf.image.crop_to_bounding_box(decoded_image, 55, 35, self.crop_image_size, self.crop_image_size),
            tf.float32)
        decoded_image_4d = tf.expand_dims(cropped_image, 0)
        resized_image = tf.image.resize_bilinear(decoded_image_4d, [self.resized_image_size, self.resized_image_size])
        record.input_image = tf.squeeze(resized_image, axis=[0])
        return record

    def _read_input_queue(self, filename_queue):
        """
        :return: a tensorflow node that will read images from the queue and construct training batches from them.
        """
        print(self.root_scope_name+"Setting up image reader...")
        read_input = self._read_input(filename_queue)
        num_preprocess_threads = 4
        num_examples_per_epoch = 800
        min_queue_examples = int(0.1 * num_examples_per_epoch)
        print(self.root_scope_name+"Shuffling")
        input_image = tf.train.batch([read_input.input_image],
                                     batch_size=self.batch_size,
                                     num_threads=num_preprocess_threads,
                                     capacity=min_queue_examples + 2 * self.batch_size
                                     )
        input_image = utils.process_image(input_image, 127.5, 127.5)
        return input_image

    def _generator(self, z, dims, train_phase, activation=tf.nn.relu, scope_name="generator"):
        """
        Create generator node. CNN with 'dims' convolutional layers.
        """
        N = len(dims)
        image_size = self.resized_image_size // (2 ** (N - 1))
        with tf.variable_scope(scope_name) as scope:
            W_z = utils.weight_variable([self.z_dim, dims[0] * image_size * image_size], name="W_z")
            b_z = utils.bias_variable([dims[0] * image_size * image_size], name="b_z")
            h_z = tf.matmul(z, W_z) + b_z
            h_z = tf.reshape(h_z, [-1, image_size, image_size, dims[0]])
            h_bnz = utils.batch_norm(h_z, dims[0], train_phase, scope="gen_bnz")
            h = activation(h_bnz, name='h_z')
            utils.add_activation_summary(h)

            for index in range(N - 2):
                image_size *= 2
                W = utils.weight_variable([5, 5, dims[index + 1], dims[index]], name="W_%d" % index)
                b = utils.bias_variable([dims[index + 1]], name="b_%d" % index)
                deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[index + 1]])
                h_conv_t = utils.conv2d_transpose_strided(h, W, b, output_shape=deconv_shape)
                h_bn = utils.batch_norm(h_conv_t, dims[index + 1], train_phase, scope="gen_bn%d" % index)
                h = activation(h_bn, name='h_%d' % index)
                utils.add_activation_summary(h)

            image_size *= 2
            W_pred = utils.weight_variable([5, 5, dims[-1], dims[-2]], name="W_pred")
            b_pred = utils.bias_variable([dims[-1]], name="b_pred")
            deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[-1]])
            h_conv_t = utils.conv2d_transpose_strided(h, W_pred, b_pred, output_shape=deconv_shape)
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
                W = utils.weight_variable([5, 5, dims[index], dims[index + 1]], name="W_%d" % index)
                b = utils.bias_variable([dims[index + 1]], name="b_%d" % index)
                h_conv = utils.conv2d_strided(h, W, b)
                if skip_bn:
                    h_bn = h_conv
                    skip_bn = False
                else:
                    h_bn = utils.batch_norm(h_conv, dims[index + 1], train_phase, scope="disc_bn%d" % index)
                h = activation(h_bn, name="h_%d" % index)
                utils.add_activation_summary(h)

            shape = h.get_shape().as_list()
            image_size = self.resized_image_size // (2 ** (N - 2))  # dims has input dim and output dim
            h_reshaped = tf.reshape(h, [self.batch_size, image_size * image_size * shape[3]])
            W_pred = utils.weight_variable([image_size * image_size * shape[3], dims[-1]], name="W_pred")
            b_pred = utils.bias_variable([dims[-1]], name="b_pred")
            h_pred = tf.matmul(h_reshaped, W_pred) + b_pred

        return tf.nn.sigmoid(h_pred), h_pred, h

    def _cross_entropy_loss(self, logits, labels, name="x_entropy"):
        xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar(name, xentropy)
        return xentropy

    def _get_optimizer(self, optimizer_name, learning_rate, optimizer_param):
        self.learning_rate = learning_rate
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(learning_rate, beta1=optimizer_param)
        elif optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    def _train(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
        return optimizer.apply_gradients(grads)

    def _setup_placeholder(self):
        self.train_phase = tf.placeholder(tf.bool)
        self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="z")


    def _dis_loss(self, logits_real, logits_fake):
        discriminator_loss_real = self._cross_entropy_loss(logits_real, tf.ones_like(logits_real),
                                                           name="disc_real_loss")

        discriminator_loss_fake = self._cross_entropy_loss(logits_fake, tf.zeros_like(logits_fake),
                                                           name="disc_fake_loss")
        return discriminator_loss_fake + discriminator_loss_real

    def _gan_loss(self, logits_real, logits_fake, feature_real, feature_fake, use_features=False):
        self.discriminator_loss = self._dis_loss(logits_real, logits_fake)
        gen_loss_disc = self._cross_entropy_loss(logits_fake, tf.ones_like(logits_fake), name="gen_disc_loss")
        if use_features:
            gen_loss_features = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (self.crop_image_size ** 2)
        else:
            gen_loss_features = 0
        self.gen_loss = gen_loss_disc + 0.1 * gen_loss_features

        tf.summary.scalar("Discriminator_loss", self.discriminator_loss)
        tf.summary.scalar("Generator_loss", self.gen_loss)

    def leaky_relu(self, x, name="leaky_relu"):
            return utils.leaky_relu(x, alpha=0.2, name=name)

    def create_network(self, generator_dims, discriminator_dims, optimizer="Adam", learning_rate=2e-4,
                       optimizer_param=0.9, improved_gan_loss=True):
        print(self.root_scope_name+"Setting up model...")
        self._setup_placeholder()
        tf.summary.histogram("z", self.z_vec)
        self.gen_images = self._generator(self.z_vec, generator_dims, self.train_phase, scope_name="generator")

        tf.summary.image("image_real", self.training_batch_images, max_outputs=2)
        tf.summary.image("image_generated", self.gen_images, max_outputs=2)

        print(self.root_scope_name+"Creating discriminator for real images")
        discriminator_real_prob, self.logits_real, feature_real = self._discriminator(self.training_batch_images, discriminator_dims,
                                                                                 self.train_phase,
                                                                                 activation=self.leaky_relu,
                                                                                 scope_name="discriminator",
                                                                                 scope_reuse=False)

        print(self.root_scope_name+"Creating discriminator for generated images")
        discriminator_fake_prob, self.logits_fake, feature_fake = self._discriminator(self.gen_images, discriminator_dims,
                                                                                 self.train_phase,
                                                                                 activation=self.leaky_relu,
                                                                                 scope_name="discriminator",
                                                                                 scope_reuse=True)

        # utils.add_activation_summary(tf.identity(discriminator_real_prob, name='disc_real_prob'))
        # utils.add_activation_summary(tf.identity(discriminator_fake_prob, name='disc_fake_prob'))

        # Loss calculation
        self._gan_loss(self.logits_real, self.logits_fake, feature_real, feature_fake, use_features=improved_gan_loss)

        train_variables = tf.trainable_variables()

        for v in train_variables:
            # print (v.op.name)
            utils.add_to_regularization_and_summary(var=v)

        self.generator_variables = [v for v in train_variables if v.name.startswith(self.root_scope_name+"generator")]
        # print(map(lambda x: x.op.name, generator_variables))
        self.discriminator_variables = [v for v in train_variables if v.name.startswith(self.root_scope_name+"discriminator")]
        # print(map(lambda x: x.op.name, discriminator_variables))

        optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

        self.generator_train_op = self._train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, optim)

    def initialize_network(self, logs_dir, checkpoint_file=None):
        print(self.root_scope_name+"Initializing network...")
        self.logs_dir = logs_dir
        self.sess = tf.Session()
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        if not checkpoint_file:
            ckpt = tf.train.get_checkpoint_state(self.logs_dir)
            if ckpt:
                checkpoint_file = ckpt.model_checkpoint_path
        else :
            checkpoint_file = logs_dir + "/" + checkpoint_file
        if checkpoint_file:
            self.saver.restore(self.sess, checkpoint_file)
            print(self.root_scope_name+"Model restored from file %s" % checkpoint_file)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)

    def dis_post_update(self):
        pass

    def get_feed_dict(self, train_phase=True):
                    batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                    feed_dict = {self.z_vec: batch_z, self.train_phase: train_phase}
                    return feed_dict

    def run_training_step(self, itr, feed_dict, batch_start_time):
        for critic_itr in range(self.critic_iterations):
            self.sess.run(self.discriminator_train_op, feed_dict=feed_dict)
            self.dis_post_update()

        self.sess.run(self.generator_train_op, feed_dict=feed_dict)

        if itr % 100 == 0:
            summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
            self.summary_writer.add_summary(summary_str, itr)

        if itr % 2000 == 0:
            batch_stop_time = time.time()
            duration = (batch_stop_time - batch_start_time) / 2000.0
            batch_start_time = batch_stop_time
            g_loss_val, d_loss_val = self.sess.run(
                [self.gen_loss, self.discriminator_loss], feed_dict=feed_dict)
            print(self.root_scope_name+"Time: %g, Step: %d, generator loss: %g, discriminator_loss: %g" % (duration, itr, g_loss_val, d_loss_val))

        if itr % 5000 == 0:
            self.saver.save(self.sess, self.logs_dir+ "model-%d.ckpt" % itr, global_step=itr)
        return batch_start_time

    def train_model(self, max_iterations):
        print(self.root_scope_name+"Training model...")
        start_time = time.time()
        batch_start_time = time.time()
        try:
            for itr in xrange(1, max_iterations):
                batch_start_time = self.run_training_step(itr, self.get_feed_dict(True), batch_start_time)
        except tf.errors.OutOfRangeError:
            print(self.root_scope_name+'Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print(self.root_scope_name+"Ending Training...")
        finally:
            print(self.root_scope_name+"Total training time: %g" % (time.time()-start_time))
            self.coord.request_stop()
            self.coord.join(self.threads)  # Wait for threads to finish.

    def visualize_model(self, logdir=None):
        if not logdir:
            logdir = self.logs_dir
        print(self.root_scope_name+"Sampling images from model...")
        batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
        feed_dict = {self.z_vec: batch_z, self.train_phase: False}

        images = self.sess.run(self.gen_images, feed_dict=feed_dict)
        images = utils.unprocess_image(images, 127.5, 127.5).astype(np.uint8)
        shape = [4, self.batch_size // 4]
        utils.save_imshow_grid(images, logdir, "generated_palette.png", shape=shape)
        for i in range(len(images)):
            scipy.misc.imsave(logdir+"/generated_image%d.png" % (i+1), images[i])

