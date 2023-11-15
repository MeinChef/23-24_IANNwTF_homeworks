import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def visualise(acc_train, acc_epoch):
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)
    
    ax1.plot(acc_epoch)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    
    ax2.plot(acc_train)
    ax2.set_xlabel('every 200th image')
    ax2.set_ylabel('accuracy')
    
    ax1.sharey(ax2)

    fig.tight_layout()

    plt.show()



def pipeline(ds):
    ds = ds.map(lambda image, label: ((tf.cast(image, tf.float32)/128)-1, label))
    ds = ds.map(lambda image, label: (tf.reshape(image, (-1,)), label))
    ds = ds.map(lambda image, label: (image, tf.one_hot(label, depth = 10)))
    ds = ds.batch(128)
    #ds = ds.prefetch(16)
    return ds

def training(model, 
             train,
             test,
             optimiser,
             loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
             epochs = 10):
    
    step_size = 20
    acc_test = np.empty(epochs)
    acc_self = np.empty((epochs, int(len(train)/20 + 1)))
    

    for epoch in range(epochs):
        counter = 0
        for x, target in train:
            
            with tf.GradientTape() as tape: #track accuracy within training
                pred = model(x)
                loss = loss_func(target, pred)

            gradients = tape.gradient(loss, model.variables) #calculate outside of the GradientTape context
            optimiser.apply_gradients(zip(gradients, model.variables))
            
            if counter % 20 == 0: 
                temp = tf.nn.softmax(pred)
                acc_self[epoch, int(counter/20)] = np.mean(np.argmax(temp, -1) == np.argmax(target.numpy, -1))
            counter += 1

        acc_test[epoch] = testing(model, test)
           
    return np.mean(acc_self, -1), acc_test


def testing(model, test):
    
    accuracy = np.zeros(len(test))
    i = 0

    for x, target in test:
        pred = model(x)
        pred = tf.nn.softmax(pred).numpy
        
        accuracy[i] = np.mean(np.argmax(pred, -1) == np.argmax(target, -1))
        print(accuracy[i])
        i += 1

    return np.mean(accuracy)
