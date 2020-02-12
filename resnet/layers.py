import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Layer
from tensorflow.keras.regularizers import l2


class ResNetLayer(Layer):
    def __init__(self, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_norm=True):
        super(ResNetLayer, self).__init__()
        self.conv1 = Conv2D(num_filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))
        if batch_norm:
            self.norm = BatchNormalization()
        else:
            self.norm = tf.identity
        self.activation = Activation(activation)

    def call(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        return self.activation(x)


class ResNetBlock(Layer):
    def __init__(self, num_filters, strides, add_projection=False):
        super(ResNetBlock, self).__init__()
        self.rnet1 = ResNetLayer(num_filters=num_filters,
                                 strides=strides)
        self.rnet2 = ResNetLayer(num_filters=num_filters,
                                 strides=1,
                                 activation=None)
        self.add = Add()
        self.activation = Activation('relu')

        if add_projection:
            self.projection = ResNetLayer(num_filters=num_filters,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_norm=False)
        else:
            self.projection = tf.identity

    def call(self, x):
        y = self.rnet1(x)
        y = self.rnet2(y)
        x = self.projection(x)
        y = self.add([x, y])
        return self.activation(y)


class ResNetStack(Layer):
    def __init__(self, num_filters, strides, first_stack=False):
        super(ResNetStack, self).__init__()

        self.rblock1 = ResNetBlock(num_filters=num_filters, strides=strides,
                                   add_projection=(not first_stack))
        self.rblock2 = ResNetBlock(num_filters=num_filters, strides=1)
        self.rblock3 = ResNetBlock(num_filters=num_filters, strides = 1)

    def call(self, x):
        x = self.rblock1(x)
        x = self.rblock2(x)
        return self.rblock3(x)
