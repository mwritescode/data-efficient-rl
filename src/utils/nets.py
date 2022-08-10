import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Rescaling, Add
from tensorflow_addons.layers import NoisyDense

INPUT_SHAPE = (4, 84,84)

def get_network(type='dqn'):
    is_noisy = 'noisy' in type
    if 'dqn' in type:
        net = _get_dqn_network(is_noisy)
    elif 'dueling' in type:
        net = _get_dueling_network(is_noisy)
    return net

def _get_dueling_network(is_noisy=False):
    dense_layer = NoisyDense if is_noisy else Dense
    model = NoisyModel if is_noisy else keras.Model

    inputs = Input(shape=INPUT_SHAPE)
    out = _get_backbone(inputs)

    value = dense_layer(units=512, activation='relu')(out)
    value = dense_layer(units=1)(value)

    advantage = dense_layer(units=512, activation='relu')(out)
    advantage = dense_layer(units=9)(advantage)

    q_value = Add()([value, advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)])
    return model(inputs=inputs, outputs=q_value, name='DuelingNetwork')

def _get_dqn_network(is_noisy=False):
    dense_layer = NoisyDense if is_noisy else Dense
    model = NoisyModel if is_noisy else keras.Model

    inputs = Input(shape=INPUT_SHAPE)
    out = _get_backbone(inputs)

    out = dense_layer(units=512, activation='relu')(out)
    out = dense_layer(units=9)(out)

    return model(inputs=inputs, outputs=out, name='DQNNetwork')

def _get_backbone(inputs):
    out = Rescaling(scale=1./255)(inputs)
    out = Conv2D(input_shape=INPUT_SHAPE, filters=32, kernel_size=8, strides=4, activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(out)

    out = Flatten()(out)
    return out


class NoisyModel(keras.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def reset_noise(self):
        dense_layers = [layer for layer in self.layers if 'dense' in layer.name.lower()]
        for layer in dense_layers:
            layer.reset_noise()