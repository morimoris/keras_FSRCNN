import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Conv2DTranspose

from keras import backend as K
from keras import constraints, initializers, regularizers
from keras.engine.base_layer import Layer


class MyPReLU(Layer): #PReLUを独自に作成
    def __init__(self,
                alpha_initializer = 'zeros',
                alpha_regularizer = None,
                alpha_constraint = None,
                shared_axes = None,
                **kwargs):
        super(MyPReLU, self).__init__(**kwargs)

        self.alpha_initializer = initializers.get('zeros')
        self.alpha_regularizer = regularizers.get(None)
        self.alpha_constraint = constraints.get(None)

    def build(self, input_shape):
        param_shape = tuple(1 for i in range(len(input_shape) - 1)) + input_shape[-1:]
        self.alpha = self.add_weight(shape = param_shape,
                                     name = 'alpha',
                                     initializer = self.alpha_initializer,
                                     regularizer = self.alpha_regularizer,
                                     constraint = self.alpha_constraint)
        self.built = True

    def call(self, inputs, mask=None):
        pos = K.relu(inputs)
        neg = -self.alpha * K.relu(-inputs)
        return pos + neg

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            }
        base_config = super(MyPReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def FSRCNN(d, s, m, mag): 
    """
    d : The LR feature dimension
    s : The number of shrinking filters
    m : The mapping depth
    mag : Magnification
    """
    input_shape = Input((None, None, 1))
    #Feature extraction
    conv2d_0 = Conv2D(filters = d,
                        kernel_size = (5, 5),
                        padding = "same",
                        activation = MyPReLU()
                        )(input_shape)
    #Shrinking
    conv2d_1 = Conv2D(filters = s,
                        kernel_size = (1, 1),
                        padding = "same",
                        activation = MyPReLU()
                        )(conv2d_0)
    #Mapping 
    conv2d_2 = conv2d_1
    for i in range(m):
        conv2d_2 = Conv2D(filters = s,
                        kernel_size = (3, 3),
                        padding = "same",
                        activation = MyPReLU()
                        )(conv2d_2)
    #Expanding
    conv2d_3 = Conv2D(filters = d,
                        kernel_size = (1, 1),
                        padding = "same",
                        activation = MyPReLU()
                        )(conv2d_2)
    #Deconvolution
    conv2d_4 = Conv2DTranspose(filters = 1,
                        kernel_size = (9, 9),
                        strides = (mag, mag),
                        padding = "same"
                        )(conv2d_3)

    model = Model(inputs = input_shape, outputs = conv2d_4)

    model.summary()

    return model

