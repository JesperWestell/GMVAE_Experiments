import tensorflow as tf
from tensorbayes.layers import gaussian_sample
from tensorbayes.distributions import log_bernoulli_with_logits, log_normal, log_squared_loss
import numpy as np

# vae subgraphs
def qy_graph(x, k=10):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qy')) > 0
    # -- q(y)
    with tf.variable_scope('qy'):
        h1 = tf.contrib.layers.fully_connected(x, 2, scope='layer1',
                                                     activation_fn=tf.nn.relu,
                                                     reuse=reuse)
        qy_logit = tf.contrib.layers.fully_connected(h1, k, scope='logit',
                                               activation_fn=tf.nn.relu, reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def qz_graph(x, y):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qz')) > 0
    # -- q(z)
    with tf.variable_scope('qz'):
        xy = tf.concat((x, y), 1, name='xy/concat')
        h1 = tf.contrib.layers.fully_connected(xy, 4, scope='layer1',
                                               activation_fn=tf.nn.relu,
                                               reuse=reuse)
        zm = tf.contrib.layers.fully_connected(h1, 2, scope='zm',
                                               activation_fn=None,
                                               reuse=reuse)
        zv = tf.contrib.layers.fully_connected(h1, 2, scope='zv',
                                               activation_fn=tf.nn.softplus,
                                               reuse=reuse)
        z = gaussian_sample(zm, zv, 'z')

        # Used to feed into z when sampling
        z = tf.identity(z,name='z_sample')
    return z, zm, zv

def z_graph(zm,zv):
    with tf.variable_scope('z'):
        z = gaussian_sample(zm, zv, 'z')
        # Used to feed into z when sampling
        z = tf.identity(z, name='z_sample')
    return z

def pz_graph(y):
    reuse = len(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pz')) > 0
    # -- p(z)
    with tf.variable_scope('pz'):
        zm = tf.contrib.layers.fully_connected(y, 2, scope='zm',
                                               activation_fn=None, reuse=reuse)
        zv = tf.contrib.layers.fully_connected(y, 2, scope='zv',
                                               activation_fn=tf.nn.softplus,
                                               reuse=reuse)
    return y, zm, zv

def px_fixed_graph(z):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px_fixed')) > 0
    # -- p(x)
    with tf.variable_scope('px_fixed'):
        h = tf.contrib.layers.fully_connected(z, 2, scope='layer1',
                                    activation_fn=tf.nn.relu,
                                    reuse=reuse,
                                    weights_initializer=tf.constant_initializer(
                                            [[1,-2],[1,-2]], verify_shape=True))
        px_logit = tf.contrib.layers.fully_connected(h, 2, scope='output',
                                                     activation_fn=None,
                                                     reuse=reuse,
                                                     weights_initializer=tf.constant_initializer(
                                                         [[1,-4],[-3,2]],
                                                         verify_shape=True))
        #px_logit = tf.identity(px_logit,name='x')
    return px_logit

def px_graph(z):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px')) > 0
    # -- p(x)
    with tf.variable_scope('px'):
        h = tf.contrib.layers.fully_connected(z, 2, scope='layer1',
                                    activation_fn=tf.nn.relu,
                                    reuse=reuse)
        px_logit = tf.contrib.layers.fully_connected(h, 2, scope='output',
                                                     activation_fn=None,
                                                     reuse=reuse)
        #px_logit = tf.identity(px_logit,name='x')
    return px_logit


def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = log_squared_loss(x, px_logit)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    return xy_loss - np.log(0.1)
