from keras.layers import *
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.backend import expand_dims

def conv2d(layer_input, filters, f_size=4, bn=False):
    """Layers used during downsampling"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d

def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
#     u = BatchNormalization(momentum=0.8)(u)
    return u
        
def create_model(category='Dense', encoding_dim=32, input_shape = 28):
    input_img = Input(shape=(input_shape, input_shape,), name='Input')
    def encoded(x):
        return None
    def decoded(x):
        return None
    
    if category=='Dense':
        def encoded(x):
            flatten = Flatten()(x)
            encoded = Dense(encoding_dim, activation='relu', name='Encoder')(flatten)
            return encoded
        def decoded(x):
            dense = Dense(input_shape * input_shape, activation='sigmoid', name='Decoder')(x)
            decoded = Reshape((input_shape, input_shape))(dense)
            return decoded
    if category=='CNN':
        gf = 5
        def encoded(x):
            x = Reshape((input_shape, input_shape, 1))(input_img)
            d1 = conv2d(x, gf)
            d2 = conv2d(d1, gf)
            d3 = Flatten()(d2)
            encoded = Dense(encoding_dim, activation='relu', name='Encoder')(d3)
            return encoded

        def decoded(x):
            u4 = Dense((input_shape * input_shape)//16 * gf, activation='relu')(x)
            u5 = Reshape((input_shape//4, input_shape//4, gf))(u4)
            u6 = deconv2d(u5, gf)
            u7 = deconv2d(u6, gf)
            u8 = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='tanh')(u7)
            u9 = Reshape((input_shape, input_shape))(u8)
            return u9

    autoencoder = Model(input_img, decoded(encoded(input_img)))
    encoder = Model(input_img, encoded(input_img))
    encoded_input = Input(shape=(encoding_dim,))
    decoder = Model(encoded_input, decoded(encoded_input))
    autoencoder.compile(optimizer='adam', loss='mse')
    return encoder, decoder, autoencoder