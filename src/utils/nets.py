from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Rescaling

def get_network(type='dqn'):
    if type == 'dqn':
        net = _get_dqn_network()
    return net

def _get_dqn_network():
    input_shape = (4, 84,84)
    inputs = Input(shape=input_shape)
    out = Rescaling(scale=1./255)(inputs)
    out = Conv2D(input_shape=input_shape, filters=32, kernel_size=8, strides=4, activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(out)

    out = Flatten()(out)
    out = Dense(units=512, activation='relu')(out)
    out = Dense(units=9)(out)

    return keras.Model(inputs=inputs, outputs=out, name='DQNNetwork')