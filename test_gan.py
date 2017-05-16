import tensorflow as tf
import numpy as np
import utils
import scipy
import matplotlib.pyplot as plt

def find_good_images(gan, scope_name, generator_dims, discriminator_dims, logs_dir, checkpoint_file):
    with tf.variable_scope(scope_name):
        with tf.Session() as sess:
            gan.initialize_network(logs_dir, checkpoint_file, sess)
            image_shape = [gan.batch_size, 64, 64, 3]
            test_image = tf.get_variable('test_image',
                                         #shape=[gan.batch_size, 64, 64, 3],
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform(image_shape, -1, 1))
            test_gan_conf, test_gan_logits, _ = gan._discriminator(test_image, discriminator_dims,
                                         gan.train_phase,
                                         activation=gan.leaky_relu,
                                         scope_name="discriminator",
                                         scope_reuse=True)
            with tf.variable_scope('test_optimizer'):
                optimizer = tf.train.AdamOptimizer()
                loss_op = -test_gan_logits[0]
                training_op = optimizer.minimize(loss_op, var_list=[test_image])
            optimizer_vars = [var for var in tf.global_variables() if 'test_optimizer' in var.name]
            print(optimizer_vars)
            test_vars_init_op = tf.variables_initializer(optimizer_vars+[test_image])
            sess.run(test_vars_init_op)
            uninitialized_vars = []
            for var in tf.global_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)

            init_new_vars_op = tf.variables_initializer(uninitialized_vars)
            sess.run(init_new_vars_op)
            #sess.run(tf.variables_initializer([gan_test]))
            sigmas = [float(i) / 100 for i in range(50)]
            confs = [.0 for _ in range(50)]
            local_confs = [.0 for _ in range(50)]
            trials_num = 10
            plt.xlabel('Variance')
            plt.ylabel('Confidence')

            for trial in range(trials_num):
                sess.run(test_vars_init_op)
                confidence = 0
                init_image = sess.run(test_image)
                for i in range(100):
                    sess.run(training_op, feed_dict={gan.train_phase: True})
                    confidence, loss = sess.run([test_gan_conf, loss_op], feed_dict={gan.train_phase: True})
                    confidence = confidence[0]
                    print(confidence, loss)
                    if confidence > 0.99:
                        break
                learned_image = sess.run(test_image)
                for i, sigma in enumerate(sigmas):
                    sess.run(test_image.assign(learned_image+np.random.normal(0.0, sigma, size=image_shape)))
                    changed_conf = sess.run(test_gan_conf, feed_dict={gan.train_phase: True})[0]
                    confs[i] += changed_conf
                    local_confs[i] = changed_conf
                    if i % 5 == 0:
                        print('Confidence using slightly modified image: %f' % changed_conf)
                image = utils.unprocess_image(learned_image, 127.5, 127.5)
                plt.plot(sigmas, local_confs, color='c', alpha=0.2)
                casted_image = image.astype(np.uint8)
                scipy.misc.imsave(logs_dir+"/learned_images_gan2/trial_%d_conf_%d.png" % (trial+1, int(confidence*1000)), casted_image[0])
            confs = [conf / trials_num for conf in confs]
            plt.plot(sigmas, confs, color='c')
            plt.savefig(logs_dir+"/learned_images_gan2/changed_image_confidence.png")

