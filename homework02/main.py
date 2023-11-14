import tensorflow as tf
import tensorflow_datasets as tfds
import mlp 
import matplotlib.pyplot as plt
import func


if __name__ == "__main__":
    (train_ds, test_ds), ds_info = tfds.load ('mnist' , split =['train', 'test'], as_supervised = True, with_info = True, shuffle_files = True)
    
    # print(ds_info)
    # How many training/test images are there?: training(60000), test(10000)
    # What's the image shape?: (28, 28, 1) -> (pixel, pixel, white value)
    # What range are pixel values in?: 0-255
    # tfds.show_examples (train_ds , ds_info)

    train = func.pipeline(train_ds)
    test = func.pipeline(test_ds)
    
    ann = mlp.MLP_Model(layer_sizes = [256, 256])
    optimiser = tf.keras.optimizers.legacy.SGD(learning_rate = 0.1)
    epochs = 50

    acc = func.training(ann, train, test, optimiser)

    func.visualise(acc)