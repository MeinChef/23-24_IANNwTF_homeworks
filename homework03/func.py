import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

@tf.function
def load_and_prep_cifar(batch_size):
    train, test = tfds.load('cifar10', split = ['train', 'test'], as_supervised = True, shuffle_files = True)

    def preprocessing(img, label):
        img = tf.cast(img, tf.float32)
        img = (img/128) - 1
        label = tf.one_hot(label, depth = 10)
        return img, label
    
    train = train.map(lambda img, label: preprocessing(img, label), num_parallel_calls = tf.data.AUTOTUNE)
    test  =  test.map(lambda img, label: preprocessing(img, label), num_parallel_calls = tf.data.AUTOTUNE)

    train = train.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test  =  test.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train, test


#@tf.function
def train_loop(model, optimizer, loss_f, train_data, test_data, num_epochs, loss_m, acc_m):

    #train_acc = tf.TensorArray(dtype = tf.float32, size = num_epochs, name = 'train_acc')
    #train_loss = tf.TensorArray(dtype = tf.float32, size = num_epochs, name = 'train_loss')
    #test_acc = tf.TensorArray(dtype = tf.float32, size = num_epochs, name = 'test_acc')
    #test_loss = tf.TensorArray(dtype = tf.float32, size = num_epochs, name = 'train_loss')


    for epoch in tf.range(num_epochs):

        tf.print(f'Epoch {epoch}:')

        for x, target in train_data:
        
            with tf.GradientTape() as tape:
                pred = model(x)
                loss = loss_f(target, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
            loss_m.update_state(loss)
            acc_m.update_state(target, pred)
        
        tf.print(f'train-{loss_m.name}: {loss_m.result().numpy()}, train-{acc_m.name}: {acc_m.result().numpy()}')
        
        #train_acc.write(epoch, acc_m.result())
        #train_loss.write(epoch, loss_m.result())

        loss_m.reset_states()
        acc_m.reset_state()

        # testing the model
        for x, target in test_data:

            pred = model(x)
            loss = loss_f(target, pred)
        
            loss_m.update_state(loss)
            acc_m.update_state(target, pred)

        tf.print(f'test-{loss_m.name}: {loss_m.result().numpy()}, test-{acc_m.name}: {acc_m.result().numpy()}')

        #test_acc.write(epoch, acc_m.result())
        #test_loss.write(epoch, loss_m.result())

        loss_m.reset_states()
        acc_m.reset_state()     

        tf.print()

        
    return #train_acc, train_loss, test_acc, test_loss




# Creating CNN 1
def this_cnn(name = "Purr"):
    inputs = tf.keras.Input(shape = (32, 32, 3), dtype = tf.float32)
    
    x = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = tf.nn.relu)(inputs)    # shape: [batch_size, 32, 32, 16]
    x = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = tf.nn.relu)(x)         # shape: [batch_size, 32, 32, 16]
    x = tf.keras.layers.MaxPooling2D()(x)                                                                           # shape: [batch_size, 16, 16, 16]

    for _ in tf.range(4):
        x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = tf.nn.relu)(x)     # shape: [batch_size, 16, 16, 32]
    x = tf.keras.layers.MaxPooling2D()(x)                                                                           # shape: [batch_size, 8, 8, 32]

    for _ in tf.range(4):
        x = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = tf.nn.relu)(x)     # shape: [batch_size, 8, 8, 64]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)                                                                 # shape: [batch_size, 64]

    outputs = tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax)(x)

    return tf.keras.Model(inputs = inputs, outputs = outputs, name = name)
    

# creating CNN 2
def that_cnn(name = "Meow"):

    inputs = tf.keras.Input(shape = (32, 32, 3), dtype = tf.float32)

    x = inputs
    for i in tf.range(12):  
        ########################
        # that would result in the first layer having 2 filters and the last layer 32768 filters - do we really want to do this? 
        # or just cut off after 11 (range 12) steps, 4069 filter
        #######################


        # since we're reducing the image size with padding = 'valid' and kernel_size = 3 by exactly 1 pixel in each direction (2 in total), 
        # we have to run it through 16 layers to be at an image size of 0. But since one can't calculate with that, we run only 11 layers
        # on it, and in the last layer global-pool-average it to reduce the 2x2 image to a vector 
        x = tf.keras.layers.Conv2D(filters = 2**(i+1), kernel_size = 3, padding = 'valid', activation = tf.nn.relu)(x)   # shape: [batch_size, 30, 30, (powers of 2)] 
        # or start with i+3 and stop at step 9 (range(10))

        if i == 11: tf.print(2**(i+1))


    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    outputs = tf.keras.layers.Dense(10, activation = tf.nn.softmax)(x)

    return tf.keras.Model(inputs = inputs, outputs = outputs, name = name)

