import tensorflow as tf
import numpy as np

from models.gan import GAN
from models.wgan import WasserstienGAN
from six.moves import xrange
import time


def s_to_hms(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def run(z_dim, crop_image_size, resized_image_size, batch_size, data_dir, generator_dims, discriminator_dims, optimizer,
        learning_rate, optimizer_param, logs_dir, checkpoint_file, max_iterations):
    print("Running discriminator cross validation")
    with tf.variable_scope('GAN'):
        gan = GAN(z_dim, crop_image_size, resized_image_size, batch_size, data_dir, root_scope_name='GAN/', critic_iterations=1)
        gan.create_network(generator_dims, discriminator_dims, optimizer, learning_rate, optimizer_param)
    with tf.variable_scope('WGAN'):
        wgan = WasserstienGAN(z_dim, crop_image_size, resized_image_size, batch_size, data_dir, root_scope_name='WGAN/', critic_iterations=25)
        wgan.create_network(generator_dims, discriminator_dims, optimizer, learning_rate, optimizer_param)
    with tf.variable_scope('GAN'):
        _, gan_dis_wgan_gen, _ = gan._discriminator(wgan.gen_images, discriminator_dims,
                                                    gan.train_phase,
                                                    activation=gan.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=True)
        gan_dis_wgan_gen_loss = gan._dis_loss(gan.logits_real, gan_dis_wgan_gen)
        tf.summary.scalar('gan_dis_wgan_gen', gan_dis_wgan_gen_loss, collections=['GAN/'])

    with tf.variable_scope('WGAN'):
        _, wgan_dis_gan_gen, _ = wgan._discriminator(gan.gen_images, discriminator_dims,
                                                    wgan.train_phase,
                                                    activation=wgan.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=True)
        wgan_dis_gan_gen_loss = wgan._dis_loss(wgan.logits_real, wgan_dis_gan_gen)
        tf.summary.scalar('wgan_dis_gan_gen', wgan_dis_gan_gen_loss, collections=['WGAN/'])

    session = tf.Session()
    with tf.variable_scope('GAN'):
        gan.initialize_network(logs_dir, checkpoint_file, session=session)
    with tf.variable_scope('WGAN'):
        wgan.initialize_network(logs_dir, checkpoint_file, session=session)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord)

    gan_time = 0.0
    wgan_time = 0.0

    def get_feed_dict(train_phase):
        gan_feed_dict = gan.get_feed_dict(train_phase)
        wgan_feed_dict = wgan.get_feed_dict(train_phase)
        merged_feed_dict = gan_feed_dict.copy()
        merged_feed_dict.update(wgan_feed_dict)
        return merged_feed_dict

    for itr in xrange(1, max_iterations):
        gan_time += gan.run_training_step(itr, get_feed_dict)
        wgan_time += wgan.run_training_step(itr, get_feed_dict)
        if itr % 2000 == 0:
            print("Step: %d, GAN time: %s, WGAN time: %s" % (itr, s_to_hms(gan_time), s_to_hms(wgan_time)))
        if itr % 5000 == 0:
            wgan.saver.save(session, logs_dir+ "/model-%d.ckpt" % itr, global_step=itr)

    coord.request_stop()
    coord.join(threads)  # Wait for threads to finish.
