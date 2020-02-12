from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D

from resnet.layers import ResNetLayer, ResNetStack


class ResNet(Model):
    def __init__(self, num_stacks=3, num_filters=16):
        super(ResNet, self).__init__()
        self.rlayer = ResNetLayer()
        self.stacks=[]
        for stack_id in range(num_stacks):
            self.stacks.append(ResNetStack(num_filters=num_filters, strides=1,
                                           first_stack=(stack_id == 0)))
            num_filters *= 2
        self.pool = AveragePooling2D(pool_size=8)
        self.flatten = Flatten()
        self.softmax = Dense(10,
                             activation='softmax',
                             kernel_initializer='he_normal')

    def call(self, x):
        x = self.rlayer(x)
        for rstack in self.stacks:
            x = rstack(x)
        x = self.pool(x)
        x= self.flatten(x)

        return self.softmax(x)