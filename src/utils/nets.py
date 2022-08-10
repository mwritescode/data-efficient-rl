import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Rescaling, Add

INPUT_SHAPE = (4, 84,84)

def get_network(type='dqn'):
    if type == 'dqn':
        net = _get_dqn_network()
    elif type == 'dueling':
        net = _get_dueling_network()
    return net

def _get_dueling_network():
    inputs = Input(shape=INPUT_SHAPE)
    out = _get_backbone(inputs)

    value = Dense(units=512, activation='relu')(out)
    value = Dense(units=1)(value)

    advantage = Dense(units=512, activation='relu')(out)
    advantage = Dense(units=9)(advantage)

    q_value = Add()([value, advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)])
    return keras.Model(inputs=inputs, outputs=q_value, name='DuelingNetwork')

def _get_backbone(inputs):
    out = Rescaling(scale=1./255)(inputs)
    out = Conv2D(input_shape=INPUT_SHAPE, filters=32, kernel_size=8, strides=4, activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(out)

    out = Flatten()(out)
    return out

def _get_dqn_network():
    inputs = Input(shape=INPUT_SHAPE)
    out = _get_backbone(inputs)
    out = Dense(units=512, activation='relu')(out)
    out = Dense(units=9)(out)

    return keras.Model(inputs=inputs, outputs=out, name='DQNNetwork')