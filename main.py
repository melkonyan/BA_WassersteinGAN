from __future__ import print_function

from scipy._lib.six import xrange

__author__ = "shekkizh"
"""
Tensorflow implementation of Wasserstein GAN
"""
import numpy as np
import tensorflow as tf
from models.gan import GAN
from models.wgan import WasserstienGAN

np.random.seed(42)
tf.set_random_seed(42)
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/CelebA_GAN_logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/CelebA_faces/", "path to dataset")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for Adam optimizer / decay for RMSProp")
tf.flags.DEFINE_float("iterations", "1e5", "No. of iterations to train model")
tf.flags.DEFINE_string("image_size", "108,64", "Size of actual images, Size of images to be generated at.")
tf.flags.DEFINE_integer("model", "0", "Model to train. 0 - GAN, 1 - WassersteinGAN")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer to use for training")
tf.flags.DEFINE_integer("gen_dimension", "16", "dimension of first layer in generator")
tf.flags.DEFINE_string("mode", "train", "train / visualize model")
tf.flags.DEFINE_string("checkpoint_file", None, "Name a file with all the variables. Should be in the logs directory.")
tf.flags.DEFINE_string("test_image", None, "Test image to be judged by a discriminator")

def main(argv=None):
    gen_dim = FLAGS.gen_dimension
    generator_dims = [64 * gen_dim, 64 * gen_dim // 2, 64 * gen_dim // 4, 64 * gen_dim // 8, 3]
    discriminator_dims = [3, 64, 64 * 2, 64 * 4, 64 * 8, 1]

    crop_image_size, resized_image_size = map(int, FLAGS.image_size.split(','))
    if FLAGS.model == 0:
        scope_name = 'GAN'
        with tf.variable_scope(scope_name):
            model = GAN(FLAGS.z_dim, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir, critic_iterations=1, root_scope_name='GAN/')
            model.create_network(generator_dims, discriminator_dims, FLAGS.optimizer, FLAGS.learning_rate,
                         FLAGS.optimizer_param)
    elif FLAGS.model == 1:
        scope_name = 'WGAN'
        with tf.variable_scope(scope_name):
            model = WasserstienGAN(FLAGS.z_dim, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir,
                           clip_values=(-0.01, 0.01), critic_iterations=25, root_scope_name='WGAN/')
            model.create_network(generator_dims, discriminator_dims, FLAGS.optimizer, FLAGS.learning_rate,
                         FLAGS.optimizer_param)
    else:
        raise ValueError("Unknown model identifier - FLAGS.model=%d" % FLAGS.model)


    if FLAGS.mode == "train":
        with tf.variable_scope(scope_name):
            model.initialize_network(FLAGS.logs_dir, FLAGS.checkpoint_file)
        model.train_model(int(1 + FLAGS.iterations))
    elif FLAGS.mode == "visualize":
        with tf.variable_scope(scope_name):
            model.initialize_network(FLAGS.logs_dir, FLAGS.checkpoint_file)
        model.visualize_model()
    elif FLAGS.mode == 'test':
        with tf.variable_scope(scope_name):
            fake_prob = model.run_dis(FLAGS.test_image, generator_dims, discriminator_dims, FLAGS.optimizer, FLAGS.learning_rate,
                             FLAGS.optimizer_param)
            model.initialize_network(FLAGS.logs_dir, FLAGS.checkpoint_file)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(model.sess, coord)
        #model.run_training_step(1, model.get_feed_dict)
        result = model.sess.run(fake_prob, feed_dict=model.get_feed_dict(True), )
        #print('%d of %d samples are considered real' % (result, model.batch_size))
        print(result)
        coord.request_stop()
        coord.join(threads)  # Wait for threads to finish.
    #import cross_dis
    #cross_dis.run(FLAGS.z_dim, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir,
    #              generator_dims, discriminator_dims, FLAGS.optimizer, FLAGS.learning_rate, FLAGS.optimizer_param,
    #              FLAGS.logs_dir, FLAGS.checkpoint_file, int(FLAGS.iterations))

if __name__ == "__main__":
    tf.app.run()
