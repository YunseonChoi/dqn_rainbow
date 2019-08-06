import tensorflow as tf
import numpy as np

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]



def mlp(x, hidden_size, act_fn=tf.tanh, name=None):

    out = tf.layers.dense(x, hidden_size, act_fn,
                          kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hidden_size)),
                          name=name)
    return out

def conv_1d(x, filters, kernel_size,
            strides=1, padding='valid', act_fn=tf.nn.leaky_relu):

    out = tf.layers.conv1d(x, filters, kernel_size, strides, padding, activation=act_fn,
                     kernel_initializer=tf.contrib.layers.xavier_initializer())
    return out

def conv_2d(x, filters, kernel_size, strides=1, padding='valid', act_fn=tf.nn.leaky_relu):

    out = tf.layers.conv2d(x, filters, kernel_size, strides, padding, activation=act_fn,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())
    return out


def conv_1d_layer(x, filters, kernel_size,
            strides=1, padding='valid', act_fn=tf.nn.leaky_relu, train=True, rate=0.1):

    out = tf.layers.conv1d(x, filters, kernel_size, strides, padding,
                     kernel_initializer=tf.contrib.layers.xavier_initializer())

    out = tf.layers.batch_normalization(out, training=train)
    out = act_fn(out)
    out = tf.layers.dropout(out, rate=rate, training=train)
    return out


def conv_2d_layer(x, filters, kernel_size,
                  strides=1, padding='valid', act_fn=tf.nn.leaky_relu, train=True, rate=0.1):

    out = tf.layers.conv2d(x, filters, kernel_size, strides, padding,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())

    out = tf.layers.batch_normalization(out, training=train)
    out = act_fn(out)
    out = tf.layers.dropout(out, rate=rate, training=train)
    return out

def transpose_conv_1d(x, filters, kernel_size, strides=1, padding='valid', act_fn=tf.nn.leaky_relu):
    out = tf.layers.conv2d_transpose(x, filters, kernel_size, strides, padding, activation=act_fn)
    return out

def transpose_conv_2d(x, filters, kernel_size, strides=1, padding='valid', act_fn=tf.nn.leaky_relu):
    x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides, padding, activation=act_fn)
    return x

def transpose_conv_1d_layer(x, filters, kernel_size, strides=1, padding='valid', act_fn=tf.nn.leaky_relu, train=True, rate=0.1):
    out = tf.layers.conv1d_transpose(x, filters, kernel_size, strides, padding)
    out = tf.layers.batch_normalization(out, training=train)
    out = act_fn(out)
    out = tf.layers.dropout(out, rate=rate, training=train)
    return out

def transpose_conv_2d_layer(x, filters, kernel_size, strides=1, padding='valid', act_fn=tf.nn.leaky_relu, train=True, rate=0.1):
    x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides, padding, activation=act_fn)
    out = tf.layers.batch_normalization(out, training=train)
    out = act_fn(out)
    out = tf.layers.dropout(out, rate=rate, training=train)
    return out





















