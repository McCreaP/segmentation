import os
import tensorflow as tf
import numpy as np

eps = 1e-5
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model")


def weight_variable(shape, gain=2.0):
    if len(shape) == 4:
        fan_in = np.prod(shape[:3])
    elif len(shape) == 2:
        fan_in = shape[0]
    else:
        raise ValueError("Invalid weight variable rank. Expected 2 or 4, got: " + str(len(shape)))
    bound = np.sqrt(gain / fan_in)
    return tf.Variable(tf.truncated_normal(shape, stddev=bound), name='weight')


def bias_variable(shape, bias=0.1):
    return const_init_variable(shape, bias, 'bias')


def scale_variable(shape, bias=1.0):
    return const_init_variable(shape, bias, 'scale')


def offset_variable(shape, bias=0.0):
    return const_init_variable(shape, bias, 'offset')


def const_init_variable(shape, bias, name):
    initializer = tf.constant(bias, shape=shape)
    return tf.Variable(initializer, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_transpose(x, W):
    in_shape = tf.shape(x)
    filter_shape = W.get_shape()
    output_shape = [in_shape[0], 2 * in_shape[1], 2 * in_shape[2], filter_shape[3]]
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, 2, 2, 1])


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_conv_layer_shape(layer):
    height = int(layer.get_shape()[1])
    width = int(layer.get_shape()[2])
    channels = int(layer.get_shape()[3])
    return height, width, channels


class Layer:
    def create(self, signals):
        raise NotImplementedError


class ConvLayer(Layer):
    def __init__(self, filter_size, num_channels, batch_norm):
        self.filter_size = filter_size
        self.num_channels = num_channels
        self.bn = batch_norm

    def create(self, signals):
        prev_layer = signals[-1]
        height, width, in_channels = get_conv_layer_shape(prev_layer)

        W = weight_variable([self.filter_size, self.filter_size, in_channels, self.num_channels])
        preact = conv2d(prev_layer, W)
        if self.bn:
            signal = self.__batch_norm(preact)
        else:
            signal = preact
        postact = tf.nn.relu(signal)
        return postact

    def __batch_norm(self, preact):
        scale = scale_variable([self.num_channels], bias=1.0)
        offset = offset_variable([self.num_channels], bias=0.0)
        means, variance = tf.nn.moments(preact, [0, 1, 2])
        return tf.nn.batch_normalization(preact, means, variance, offset, scale, eps)


class UpConvLayer(Layer):
    def __init__(self, filter_size, num_channels, batch_norm):
        self.filter_size = filter_size
        self.num_channels = num_channels
        self.bn = batch_norm

    def create(self, signals):
        prev_layer = signals[-1]
        height, width, in_channels = get_conv_layer_shape(prev_layer)
        W = weight_variable([self.filter_size, self.filter_size, in_channels, self.num_channels])
        preact = conv2d_transpose(prev_layer, W)
        if self.bn:
            signal = self.__batch_norm(preact)
        else:
            signal = preact
        postact = tf.nn.relu(signal)
        return postact

    def __batch_norm(self, preact):
        scale = scale_variable([self.num_channels], bias=1.0)
        offset = offset_variable([self.num_channels], bias=0.0)
        means, variance = tf.nn.moments(preact, [0, 1, 2])
        return tf.nn.batch_normalization(preact, means, variance, offset, scale, eps)


class PoolLayer(Layer):
    def create(self, signals):
        return max_pool_2x2(signals[-1])


class ConcatLayer(Layer):
    def __init__(self, layer_id_to_concat):
        self.layer_id_to_concat = layer_id_to_concat

    def create(self, signals):
        return tf.concat([signals[-1], signals[self.layer_id_to_concat]], -1)


class Model:
    def __init__(self):
        # 256 x 256
        layers = [
            # Input is a first signal
            ConvLayer(filter_size=3, num_channels=20, batch_norm=False),  # 1
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),  # 3
            PoolLayer(),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),  # 7
            PoolLayer(),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),  # 11
            PoolLayer(),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),  # 15
            PoolLayer(),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),  # 19
            PoolLayer(),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            UpConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConcatLayer(19),
            ConvLayer(filter_size=3, num_channels=96, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            UpConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConcatLayer(15),
            ConvLayer(filter_size=3, num_channels=96, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            UpConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConcatLayer(11),
            ConvLayer(filter_size=3, num_channels=96, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            UpConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConcatLayer(7),
            ConvLayer(filter_size=3, num_channels=96, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            UpConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConcatLayer(3),
            ConvLayer(filter_size=3, num_channels=96, batch_norm=True),
            ConvLayer(filter_size=3, num_channels=64, batch_norm=True),
            ConvLayer(filter_size=1, num_channels=66, batch_norm=False)
        ]

        self.x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x')
        signals = [self.x]

        for layer_idx, layer in enumerate(layers):
            with tf.variable_scope("layer_" + str(layer_idx + 1)):
                signal = layer.create(signals)
                signals.append(signal)

        self.y_conv = signals[-1]
        self.y_conv_predictions = tf.reshape(
            tf.argmax(self.y_conv, axis=3),
            [-1, 256, 256, 1]
        )
