import tensorflow as tf
from utils import *
from huber_loss import *
import numpy as np



def mlp_nn(x, n_hiddens=[64, 32, 32], act_fn=tf.nn.leaky_relu):

    h = x
    for layer_i, n_hidden in enumerate(n_hiddens):
        with tf.variable_scope('mlp/layer_{}'.format(layer_i)):
            h = mlp(h, n_hidden, act_fn)

            print('mlp/layer_{}:'.format(layer_i).ljust(13, ' ') + str(h.get_shape().as_list()))

    return h

def conv_1d_nn(x, n_filters=[16, 8, 8], kernel_sizes=[5, 3, 1], strides=[2, 1, 1],
               padding='valid', act_fn=tf.nn.leaky_relu):

    assert len(n_filters) == len(kernel_sizes) == len(strides)

    h = x
    for layer_i, filter in enumerate(n_filters):
        with tf.variable_scope('conv_1d/layer_{}'.format(layer_i)):
            h = conv_1d(h, filter, kernel_sizes[layer_i], strides[layer_i], padding, act_fn)

            print('conv_1d/layer_{}:'.format(layer_i).ljust(13, ' ') + str(h.get_shape().as_list()))

    return h


def conv_2d_nn(x, n_filters=[16, 16, 8], kernel_sizes=[5, 5, 3], strides=[2, 2, 1],
               padding='valid', act_fn=tf.nn.leaky_relu):

    assert len(n_filters) == len(kernel_sizes) == len(strides)
    h = x
    for layer_i, filter in enumerate(n_filters):
        with tf.variable_scope('conv_2d/layer_{}'.format(layer_i)):
            h = conv_2d(h, filter, kernel_sizes[layer_i], strides[layer_i], padding, act_fn)
            print('conv_2d/layer_{}:'.format(layer_i).ljust(13, ' ') + str(h.get_shape().as_list()))

    return h


def q_network(x, action_size, nn_type='mlp', add_n_hiddens=[], **kwargs):
    """
    Args
     x: input, state
     action_size: int
     nn_type: string
        'mlp'. multi-layers perceptron
        'conv_1d'. 1d convolution
        'conv_1d_with_batch', 1d- > batch -> act_fn- > dropout
        'conv_2d'. 1d convolution
     n_hiddens: list
        before get action, added extra mlp layers.
        list of the number of hidden unit in the each layer

    """
    if nn_type is 'mlp':
        assert len(x.get_shape()) == 2
        h = mlp_nn(x, **kwargs)

    elif nn_type is 'conv_1d':
        assert len(x.get_shape()) == 3
        h = conv_1d_nn(x, **kwargs)
        h = tf.layers.flatten(h)
        print('flatten:'.ljust(13, ' ') + str(h.get_shape().as_list()))

    elif nn_type is 'conv_2d':
        assert len(x.get_shape()) == 4
        h = conv_2d_nn(x, **kwargs)
        h = tf.layers.flatten(h)
        print('flatten:'.ljust(13, ' ') + str(h.get_shape().as_list()))


    for layer_i, n_hidden in enumerate(add_n_hiddens):
        with tf.variable_scope('mlp_q/layer_{}'.format(layer_i)):
            h = mlp(h, n_hidden, act_fn=tf.nn.relu)
            print('mlp_q/layer_{}:'.format(layer_i).ljust(13, ' ') + str(h.get_shape().as_list()))

    with tf.variable_scope('mlp_q/q_value'):
        q = mlp(h, action_size, act_fn=None, name='q_value')
        print('mlp_q/q_value:'.ljust(13, ' ') + str(q.get_shape().as_list()))

    return q


def duel_q_network(x, action_size, nn_type='mlp', add_n_hiddens=[], **kargs):
    """
    Args
     x: input, state
     action_size: int
     nn_type: string
        'mlp'. multi-layer perceptron
        'conv_1d'. 1d convolution
        'conv_1d_with_batch', 1d- > batch -> act_fn- > dropout
        'conv_2d'. 1d convolution
     n_hiddens: list
        before get action, added extra mlp layers.
        list of the number of hidden unit in the each layer

    """
    if nn_type is 'mlp':
        assert len(x.get_shape()) == 2
        h = mlp_nn(x, **kargs)

    elif nn_type is 'conv_1d':
        assert len(x.get_shape()) == 3
        h = conv_1d_nn(x, **kargs)
        h = tf.layers.flatten(h)
        print('flatten:'.ljust(13, ' ') + str(h.get_shape().as_list()))

    elif nn_type is 'conv_2d':
        assert len(x.get_shape()) == 4
        h = conv_2d_nn(x, **kargs)
        h = tf.layers.flatten(h)
        print('flatten:'.ljust(13, ' ') + str(h.get_shape().as_list()))


    for layer_i, n_hidden in enumerate(add_n_hiddens):
        with tf.variable_scope('mlp_q/layer_{}'.format(layer_i)):
            h = mlp(h, n_hidden, act_fn=tf.nn.relu)
            print('mlp_q/layer_{}:'.format(layer_i).ljust(13, ' ') + str(h.get_shape().as_list()))

    with tf.variable_scope('mlp_q/adv_value'):
        adv = mlp(h, action_size, act_fn=None, name='adv_value')
        print('mlp_q/adv_value:'.ljust(13, ' ') + str(adv.get_shape().as_list()))

    with tf.variable_scope('mlp_q/v_value'):
        v = mlp(h, 1, act_fn=None, name='q_value')
        print('mlp_q/v_value:'.ljust(13, ' ') + str(v.get_shape().as_list()))

    q = tf.nn.relu(v + adv - tf.reduce_mean(adv), name='q_value')
    return q











