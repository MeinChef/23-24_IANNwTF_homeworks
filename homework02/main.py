import tensorflow as tf
import tensorflow_datasets as tfds
import mlp 
import func
import numpy as np


if __name__ == "__main__":
    
    
    (train_ds, test_ds), ds_info = tfds.load ('mnist' , split =['train', 'test'], as_supervised = True, with_info = True, shuffle_files = True)
    
    # print(ds_info)
    #
    # How many training/test images are there?: training(60000), test(10000)
    # What's the image shape?: (28, 28, 1) -> (pixel, pixel, white value)
    # What range are pixel values in?: 0-255
    # tfds.show_examples (train_ds , ds_info)
    
    train = func.pipeline(train_ds)
    test = func.pipeline(test_ds)
    
    ann = mlp.MLP_Model(layer_sizes = [256, 256]) # task 3: more neurons or more layers?
    optimiser = tf.keras.optimizers.legacy.SGD(learning_rate = 0.1)
    
    
    acc_self, acc_test, loss_self, loss_test = func.training_track_all_params(ann, train, test, optimiser)

    #func.visualise(acc_test, acc_self, loss_test, loss_self)
    


    ###################################
    # Altering Parameters:
    ###################################
    # learning rate                 base, 2 params: (0.1, 0.03, 0.5)
    # batchsize                     base, 2 params: (128, 32, 512)
    # no. o. layers                 base, 2 params: (2, 1, 8)
    # no. o. neurons. per layer     base, 2 params: (256, 64, 512)
    # optimiser                     base, 2 params: (sgd, lion, adam)


    learning_rates = np.array([0.03, 0.5])
    batch_sizes = np.array([32, 512])
    network_sizes = np.array([1, 8])
    layer_sizes = np.array([64, 512])   
    optimisers = [tf.keras.optimizers.Lion(), tf.keras.optimizers.Adam()] 


    acc_learn = [acc_test]
    acc_batch = [acc_test]
    acc_net = [acc_test]
    acc_layer = [acc_test]
    acc_opt = [acc_test]
    
    ann0 = mlp.MLP_Model(layer_sizes = [256, 256])

    for learning_rate, i in zip(learning_rates, range(2)): 
        acc_learn.append(func.var_learn(learning_rate, train, test, ann0))


    ann0 = mlp.MLP_Model(layer_sizes = [256, 256])
    
    for batch, i in zip(batch_sizes, range(2)): 
        acc_batch.append(func.var_batch(batch, train_ds, test_ds, ann0))

    for network_size, i in zip(network_sizes, range(2)): 
        acc_net.append(func.var_net(network_size, train, test, optimiser))
    
    for layer, i in zip(layer_sizes, range(2)):
        acc_layer.append(func.var_layer(layer, train, test, optimiser))

    for opt, i in zip(optimisers, range(2)):
        acc_opt.append(func.var_opt(opt, ann, train, test))

    accuracy_array = np.array([acc_learn, acc_batch, acc_net, acc_layer, acc_opt])

    print(accuracy_array.shape)
    func.vis_accs(accuracy_array)

