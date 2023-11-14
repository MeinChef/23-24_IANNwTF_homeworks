import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def visualise():
    pass

def pipeline(ds):
    ds = ds.map(lambda image, label: ((tf.cast(image, tf.float32)/128)-1, label))
    ds = ds.map(lambda image, label: (tf.reshape(image, (-1)), label))
    ds = ds.map(lambda image, label: (image, tf.one_hot(label, depth = 10)))
    ds = ds.batch(128)
    ds = ds.prefetch(16)
    return ds

def training(model, 
             train,
             test,
             optimiser,
             loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
             epochs = 10):
    
    accuracy = np.empty(epochs)

    for epoch in range(epochs):
        for x, target in train:
            
            with tf.GradientTape() as tape: #track accuracy within training
                pred = model(x)
                loss = loss_func(target, pred)
            gradients = tape.gradient(loss, model.variables) #calculate outside of the GradientTape context
            optimiser.apply_gradients(zip(gradients, model.variables))

        accuracy = testing(model, test)
           
    return accuracy


def testing(model, test):
    
    accuracy = np.array(shape = len(test), dtype = bool)
    i = 0

    for x, target in test:
        pred = model(x)
        pred = tf.nn.softmax(pred).numpy

        if np.argmax(pred) == np.argmax(target):
            accuracy[i] = True
        else:
            accuracy[i] = False

        i += 1

    return np.mean(accuracy)
