import tensorflow as tf
from six.moves import xrange


def s_to_hms(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def run(gan1, gan2, gan1_scope_name, gan2_scope_name, discriminator_dims, logs_dir, checkpoint_file, max_iterations):
    print("Running discriminator cross validation")
    with tf.variable_scope(gan1_scope_name):
        _, gan_dis_wgan_gen, _ = gan1._discriminator(gan2.gen_images, discriminator_dims,
                                                    gan1.train_phase,
                                                    activation=gan1.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=True)
        gan_dis_wgan_gen_loss = gan1._dis_loss(gan1.logits_real, gan_dis_wgan_gen)
        tf.summary.scalar('%s_dis_%s_gen' % (gan1_scope_name, gan2_scope_name), gan_dis_wgan_gen_loss, collections=gan1.summary_collections)

    with tf.variable_scope(gan2_scope_name):
        _, wgan_dis_gan_gen, _ = gan2._discriminator(gan1.gen_images, discriminator_dims,
                                                    gan2.train_phase,
                                                    activation=gan2.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=True)
        wgan_dis_gan_gen_loss = gan2._dis_loss(gan2.logits_real, wgan_dis_gan_gen)
        tf.summary.scalar('%s_dis_%s_gen' % (gan2_scope_name, gan1_scope_name), wgan_dis_gan_gen_loss, collections=gan2.summary_collections)

    session = tf.Session()
    with tf.variable_scope(gan1_scope_name):
        gan1.initialize_network(logs_dir, checkpoint_file, session=session)
    with tf.variable_scope(gan2_scope_name):
        gan2.initialize_network(logs_dir, checkpoint_file, session=session)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord)

    gan1_time = 0.0
    gan2_time = 0.0

    def get_feed_dict(train_phase):
        gan_feed_dict = gan1.get_feed_dict(train_phase)
        wgan_feed_dict = gan2.get_feed_dict(train_phase)
        merged_feed_dict = gan_feed_dict.copy()
        merged_feed_dict.update(wgan_feed_dict)
        return merged_feed_dict

    for itr in xrange(1, max_iterations):
        gan1_time += gan1.run_training_step(itr, get_feed_dict)
        gan2_time += gan2.run_training_step(itr, get_feed_dict)
        if itr % 2000 == 0:
            print("Step: %d, %s time: %s, %s time: %s" % (itr, gan1_scope_name, s_to_hms(gan1_time), gan2_scope_name, s_to_hms(gan2_time)))
        if itr % 5000 == 0:
            gan2.saver.save(session, logs_dir+ "/model-%d.ckpt" % itr, global_step=itr)

    coord.request_stop()
    coord.join(threads)  # Wait for threads to finish.
