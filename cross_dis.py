import tensorflow as tf
import numpy as np

tf.set_random_seed(42)
np.random.seed(42)

from models.gan import GAN
from models.wgan import WasserstienGAN
from six.moves import xrange
import time

def run(z_dim, crop_image_size, resized_image_size, batch_size, data_dir, generator_dims, discriminator_dims, optimizer,
        learning_rate, optimizer_param, logs_dir, checkpoint_file, max_iterations):
    print("Running discriminator cross validation")
    with tf.variable_scope('GAN'):
        gan = GAN(z_dim, crop_image_size, resized_image_size, batch_size, data_dir, root_scope_name='GAN/')
        gan.create_network(generator_dims, discriminator_dims, optimizer, learning_rate, optimizer_param)
    with tf.variable_scope('WGAN'):
        wgan = WasserstienGAN(z_dim, crop_image_size, resized_image_size, batch_size, data_dir, root_scope_name='WGAN/')
        wgan.create_network(generator_dims, discriminator_dims, optimizer, learning_rate, optimizer_param)
    with tf.variable_scope('GAN'):
        _, gan_dis_wgan_gen, _ = gan._discriminator(wgan.gen_images, discriminator_dims,
                                                    gan.train_phase,
                                                    activation=gan.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=True)
        gan_dis_wgan_gen_loss = gan._dis_loss(gan.logits_real, gan_dis_wgan_gen)
        tf.summary.scalar('gan_discriminator_wgan_generator', gan_dis_wgan_gen_loss)

    with tf.variable_scope('WGAN'):
        _, wgan_dis_gan_gen, _ = wgan._discriminator(gan.gen_images, discriminator_dims,
                                                    wgan.train_phase,
                                                    activation=wgan.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=True)
        wgan_dis_gan_gen_loss = wgan._dis_loss(wgan.logits_real, wgan_dis_gan_gen)
        tf.summary.scalar('gan_discriminator_wgan_generator', wgan_dis_gan_gen_loss)

    session = tf.Session()
    with tf.variable_scope('GAN'):
        gan.initialize_network(logs_dir, checkpoint_file, session=session)
    with tf.variable_scope('WGAN'):
        wgan.initialize_network(logs_dir, checkpoint_file, session=session)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord)

    gan_batch_time = time.time()
    wgan_batch_time = time.time()

    for itr in xrange(1, max_iterations):
        gan_feed_dict = gan.get_feed_dict(True)
        wgan_feed_dict = wgan.get_feed_dict(True)
        merged_feed_dict = gan_feed_dict.copy()
        merged_feed_dict.update(wgan_feed_dict)
        gan_batch_time = gan.run_training_step(itr, merged_feed_dict, gan_batch_time)
        wgan_batch_time = wgan.run_training_step(itr, merged_feed_dict, wgan_batch_time)

    coord.request_stop()
    coord.join(threads)  # Wait for threads to finish.
