import tensorflow as tf
import tensorflow_datasets as tfds
import mlp 
import matplotlib.pyplot as plt

def visualise():
    pass

if __name__ == "__main__":
    (train_ds, test_ds), ds_info = tfds.load ('mnist' , split =['train', 'test'], as_supervised = True, with_info = True, shuffle_files = True)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
