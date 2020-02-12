import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, \
    LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from resnet.models import ResNet


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def _main(filepath='.'):
    """ Run ResNet20 on cifar10 """

    cifar = tf.keras.datasets.cifar10
    model = ResNet()
    (x_train, y_train), (x_test, y_test) = cifar.load_data()

    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

    datagen.fit(x_train)

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])

    callbacks = [
        LearningRateScheduler(lr_schedule),
        ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6, verbose=1),
        ModelCheckpoint(filepath=os.path.join(filepath, 'model_checkpoint.h5'),
                        verbose=1, save_best_only=True, save_weights_only=False)
    ]
    model.fit(datagen.flow(x_train, y_train, batch_size=128),
              callbacks=callbacks,
              validation_data=(x_test, y_test),
              epochs=200)
    model.save(os.path.join(filepath, 'model_final.tf'), save_format="tf")


if __name__ == '__main__':
    _main()
