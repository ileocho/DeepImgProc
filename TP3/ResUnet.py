import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Activation, Concatenate, Input, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

"""
This file is my answer to Q1 and Q2 of the assignment for Preligens intership process.
The first question was :
    a. the number of filters can be modified :
        * The number of filters can be modified by changing the filter_size variable in the U-Net model instantiation.
    b. the sizes of the convolution kernels can be changed :
        * The size of the convolution kernels can be changed by
        * changing the kernel_size variable in the U-Net model instantiation.
    c. the convolution blocks can be replaced with ResNet blocks :
        * The convolution blocks can be replaced with ResNet blocks by
        * changing the resnet (bool) variable in the U-Net model instantiation.
        * The blocks I used are the ones from the ResNet paper, with
        * Batch Normalisation between Conv2Ds and activations (https://arxiv.org/pdf/1512.03385.pdf).

Second question was :
    a. Create a keras.Model object of only the encoder from the full model :
        * The class Encoder is created for this purpose. It takes the same arguments as
        * the U-Net model and creates a keras.Model object of only the encoder from the full model.
    b. Imagine we want to create a model that takes two images with the same
        size as inputs and feeds themindependently to two distincts encoder networks.
        Then, the two encoder outputs (feature maps) would beconcatenated, and the decoder
        network would take this concatenation as an input. The architecture isillustrated
        in figure 1. How would you create such a model in Keras ? :
        * To create such a model, we need two encoders and one decoder. The two encoders
        * are created by calling the Encoder class twice, and the decoder is created by
        * calling the U-Net class. The output of the two encoders are concatenated and
        * passed to the decoder, while the skip connections from each encoder are added together to be passed once
        * to the decoder. This type of model can be created by calling the UNet class with
        * the mtype argument set to ['double'].
        * This model would requier two inputs, one for each image.

All models compile (see main function) but I didn't try to train them with the according data.
"""

class Down(tf.Module):
    """
    Downsampling block for the U-Net.

    Args:
        filter_size (int): Number of filters in the convolutional layers
        kernel_size (int): Size of the convolutional kernels
        resnet (bool, optional): Whether to use a ResNet block. Defaults to False.
        drop (bool, optional): Whether to use dropout. Defaults to False.
        name (str, optional): Name of the block. Defaults to 'b
    """
    def __init__(self, filter_size, kernel_size, resnet=False,  drop=False, name='b'):
        self.drop_cond = drop
        self.resnet = resnet
        self.filter_size = filter_size
        self.kernel_size = kernel_size

    def conv_block(self, x):
        down = Conv2D(self.filter_size, self.kernel_size, activation='relu', padding='same')(x)
        conv = Conv2D(self.filter_size, self.kernel_size, activation='relu', padding='same')(down)
        drop = Dropout(0.5)(conv) if self.drop_cond else conv
        out = MaxPooling2D(pool_size=(2, 2))(drop)
        return out, drop

    def res_block(self, x):
        conv = Conv2D(self.filter_size, self.kernel_size, padding='same')(x)
        bn = BatchNormalization()(conv)
        act = Activation('relu')(bn)
        conv = Conv2D(self.filter_size, self.kernel_size, padding='same')(act)
        bn = BatchNormalization()(conv)
        act = Activation('relu')(bn)
        drop = Dropout(0.5)(act) if self.drop_cond else act
        residual = Conv2D(self.filter_size, 1, activation='relu', padding='same')(x)
        residual = Add()([drop, residual])
        out = MaxPooling2D(pool_size=(2, 2))(residual)
        return out, residual

    def __call__(self, x):
        if self.resnet:
            return self.res_block(x)
        else:
            return self.conv_block(x)


class Up(tf.Module):
    """
    Upsampling block for the U-Net.

    Args:
        filter_size (int): Number of filters in the convolutional layers
        kernel_size (int): Size of the convolutional kernels
        resnet (bool, optional): Whether to use a ResNet block. Defaults to False.
        name (str, optional): Name of the block. Defaults to 'b
    """
    def __init__(self, filter_size, kernel_size, resnet=False, name='b'):
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.resnet = resnet

    def conv_block(self, x, skip):
        up = UpSampling2D(size=(2, 2))(x)
        concat = Concatenate(axis=3)([up, skip])
        conv = Conv2D(self.filter_size, self.kernel_size, activation='relu', padding='same')(concat)
        conv = Conv2D(self.filter_size, self.kernel_size, activation='relu', padding='same')(conv)
        return conv

    def res_block(self, x, skip):
        up = UpSampling2D(size=(2, 2))(x)
        concat = Concatenate(axis=3)([up, skip])
        conv = Conv2D(self.filter_size, self.kernel_size, padding='same')(concat)
        bn = BatchNormalization()(conv)
        act = Activation('relu')(bn)
        conv = Conv2D(self.filter_size, self.kernel_size, padding='same')(act)
        bn = BatchNormalization()(conv)
        act = Activation('relu')(bn)
        residual = Conv2D(self.filter_size, 1, activation='relu', padding='same')(act)
        residual = Add()([conv, residual])
        return residual

    def __call__(self, x, skip):
        if self.resnet:
            return self.res_block(x, skip)
        else:
            return self.conv_block(x, skip)


class Encoder(tf.keras.Model):
    """
    Encoder for the U-Net.
    It is composed of 4 downsampling blocks. The first 3 are identical, and the last one has dropout.
    It returns the output of the last block and all skip connections in a list. The skip connections are used in the decoder.

    Args:
        filter_size (int): Number of filters in the convolutional layers
        kernel_size (int): Size of the convolutional kernels
        input_size (tuple, optional): Size of the input image. Defaults to (256,256,1).
        resnet (bool, optional): Whether to use a ResNet block. Defaults to False.
        name (str, optional): Name of the block. Defaults to 'e'
    """
    def __init__(self, filter_size, kernel_size, input_size=(256,256,1), resnet=False, name='e'):
        super(Encoder, self).__init__(name=name)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.resnet = resnet
        self.skips = {}
        self.__name__ = name

        self.down1 = Down(filter_size, kernel_size, resnet, name=name+'_d1')
        self.down2 = Down(filter_size*2, kernel_size, resnet, name=name+'_d2')
        self.down3 = Down(filter_size*4, kernel_size, resnet, name=name+'_d3')
        self.down4 = Down(filter_size*8, kernel_size, resnet, drop=True, name=name+'_d4')

    def __call__(self, x):
        down1, skip1 = self.down1(x)
        down2, skip2 = self.down2(down1)
        down3, skip3 = self.down3(down2)
        down4, skip4 = self.down4(down3)

        return down4, [skip1, skip2, skip3, skip4]


class BottleNeck(tf.keras.Model):
    """
    Bottleneck block for the U-Net.
    Composed of 2 convolutional layers with dropout.
    Args:
        filter_size (int): Number of filters in the convolutional layers
        kernel_size (int): Size of the convolutional kernels
        resnet (bool, optional): Whether to use a ResNet block. Defaults to False.
        name (str, optional): Name of the block. Defaults to 'b
    """
    def __init__(self, filter_size, kernel_size, resnet=False, name='b'):
        super(BottleNeck, self).__init__(name=name)
        self.resnet = resnet
        self.filter_size = filter_size
        self.kernel_size = kernel_size

    def bottleneck(self, x):
        conv = Conv2D(self.filter_size, self.kernel_size, activation='relu', padding='same')(x)
        conv = Conv2D(self.filter_size, self.kernel_size, activation='relu', padding='same')(conv)
        drop = Dropout(0.5)(conv)
        return drop

    def res_bottleneck(self, x):
        conv = Conv2D(self.filter_size, self.kernel_size, padding='same')(x)
        bn = BatchNormalization()(conv)
        act = Activation('relu')(bn)
        conv = Conv2D(self.filter_size, self.kernel_size, padding='same')(act)
        bn = BatchNormalization()(conv)
        act = Activation('relu')(bn)
        drop = Dropout(0.5)(act)
        residual = Conv2D(self.filter_size, 1, activation='relu', padding='same')(x)
        residual = Add()([drop, residual])
        return residual

    def __call__(self, x):
        if self.resnet:
            return self.res_bottleneck(x)
        else:
            return self.bottleneck(x)


class Decoder(tf.keras.Model):
    """
    Decoder for the U-Net.
    It is composed of 4 upsampling blocks taking in the skip connections from the encoder.
    It returns the output of the last upsampling block.

    Args:
        filter_size (int): Number of filters in the convolutional layers
        kernel_size (int): Size of the convolutional kernels
        resnet (bool, optional): Whether to use a ResNet block. Defaults to False.
        name (str, optional): Name of the block. Defaults to 'd'
    """
    def __init__(self, filter_size, kernel_size, resnet=False, name='d'):
        super(Decoder, self).__init__(name=name)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.resnet = resnet
        self.__name__ = name

        self.up1 = Up(filter_size*8, kernel_size, resnet, name=name+'_u1')
        self.up2 = Up(filter_size*4, kernel_size, resnet, name=name+'_u2')
        self.up3 = Up(filter_size*2, kernel_size, resnet, name=name+'_u3')
        self.up4 = Up(filter_size, kernel_size, resnet, name=name+'_u4')

    def __call__(self, x, skip):
        up1 = self.up1(x, skip[3])
        up2 = self.up2(up1, skip[2])
        up3 = self.up3(up2, skip[1])
        up4 = self.up4(up3, skip[0])

        return up4


class UNet(tf.keras.Model):
    """
    U-Net model.
    This U-Net has two forms: simple and double.
    The simple U-Net has a single encoder and decoder.
    The double U-Net has two encoders and one decoder.
    ---
    Simple : skip connections are passed from the encoder to the decoder.
    Double : skip connections from both encoders are added together and passed to the decoder,
             and outputs from both encoders are concatenated and passed to the bottleneck.
    ---

    Args:
        filter_size (int): Number of filters in the convolutional layers
        kernel_size (int): Size of the convolutional kernels
        input_size (tuple, optional): Size of the input image. Defaults to (256,256,1).
        resnet (bool, optional): Whether to use a ResNet block. Defaults to False.
        mtype (list, optional): Type of U-Net. Defaults to ['simple', 'double'].
        name (str, optional): Name of the block. Defaults to 'u'

    ---
    Example:

    `unet = UNet(64, 3, (256,256,1), resnet=True, mtype=['simple'], name='unet')(x))`

    `unet = UNet(64, 3, (256,256,1), resnet=False, mtype=['double'], name='double_conv_unet')(x1, x2)`

    """
    def __init__(self, filter_size, kernel_size, input_size=(256,256,1), resnet=False, mtype=['simple', 'double'], name='u'):
        super(UNet, self).__init__(name=name)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.resnet = resnet
        self.mtype = mtype
        self.__name__ = name

        self.encoder = Encoder(filter_size, kernel_size, input_size, resnet, name=name+'_e')
        self.encoder_2 = Encoder(filter_size, kernel_size, input_size, resnet, name=name+'_e2')
        self.bottleneck = BottleNeck(filter_size*16, kernel_size, resnet, name=name+'_b')
        self.decoder = Decoder(filter_size, kernel_size, resnet, name=name+'_d')
        self.conv = Conv2D(2, kernel_size, activation='sigmoid', padding='same')
        self.out = Conv2D(1, 1, activation='sigmoid', padding='same')

    def simple(self, x):
        down4, skips = self.encoder(x)
        bottleneck = self.bottleneck(down4)
        up4 = self.decoder(bottleneck, skips)

        conv = self.conv(up4)
        out = self.out(conv)

        return out

    def double(self, x1, x2):
        down4, skips = self.encoder(x1)
        down4_2, skips_2 = self.encoder_2(x2)

        down = tf.concat([down4, down4_2], axis=-1)
        skips = [tf.add(s1, s2) for s1, s2 in zip(skips, skips_2)]

        bottleneck = self.bottleneck(down)

        up4 = self.decoder(bottleneck, skips)

        conv = self.conv(up4)
        out = self.out(conv)

        return out

    def __call__(self, x1, x2=None):
        if 'simple' in self.mtype:
            return tf.keras.Model(inputs=x1, outputs=self.simple(x1))
        elif 'double' in self.mtype:
            return tf.keras.Model(inputs=[x1, x2], outputs=self.double(x1, x2))
        else:
            raise ValueError('Invalid model type, must be either "simple" or "double')