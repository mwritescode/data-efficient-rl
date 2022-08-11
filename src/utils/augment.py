import numpy as np
import tensorflow as tf

IMG_SIZE = (84,84)

def augment_batch(batch):
    batch = _random_crop(batch)
    return _random_intensity(batch)

def _random_crop(batch):
    batch = np.pad(batch, pad_width=((0,0), (0,0), (4,4), (4,4)), mode='edge')
    # Different random crop for every stakced frame 
    batch = tf.map_fn(lambda img: tf.map_fn(lambda ch: tf.image.random_crop(ch, IMG_SIZE), img), batch)

    # Same crop for every stacked frame would have been: 
    # tf.map_fn(lambda img: tf.image.random_crop(img, IMG_SIZE), batch)
    return batch

def _random_intensity(batch, scale=0.05):
    batch = tf.map_fn(lambda img: tf.map_fn(lambda ch: _adjust_intensity(ch, scale), img), batch)
    return batch

def _adjust_intensity(img, scale):
    r = np.random.normal(loc=0.0, scale=1.0)
    img = img * (1.0 + scale * np.clip(r, -2, 2))
    return img

