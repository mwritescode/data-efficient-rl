import numpy as np
import tensorflow as tf

IMG_SIZE = (4, 84,84)

@tf.function
def augment_batch(batch):
    batch = tf.pad(batch, paddings=((0,0), (0,0), (1, 1), (1,1)), mode='SYMMETRIC')
    batch = tf.pad(batch, paddings=((0,0), (0,0), (1, 1), (1,1)), mode='SYMMETRIC')
    batch = tf.pad(batch, paddings=((0,0), (0,0), (2, 2), (2,2)), mode='SYMMETRIC')

    batch = tf.cast(batch, tf.float32)

    batch = tf.map_fn(lambda img:  _augment_img(img, scale=tf.convert_to_tensor(0.05)), batch)
    return batch

def _augment_img(img, scale=0.05):
    img = tf.image.random_crop(img, IMG_SIZE)
    img = _adjust_intensity(img=img, scale=scale)
    return img

def _adjust_intensity(img, scale):
    r = tf.random.normal(shape=(1,))
    img = img * (1.0 + scale * tf.clip_by_value(r, -2.0, 2.0))
    return img

