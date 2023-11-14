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
             learning_rate = 0.1,
             loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits = True), 
             optimizer = tf.keras.optimizers.legacy.SGD(),
             epochs = 10):
    
    losses = []

    for epoch in range(epochs):
        for x, target in train:
            
            with tf.GradientTape() as tape:
                pred = model(x)
                loss = loss_func(target, pred)
            gradients = tape.gradient(loss, model.variables) #calculate outside of the GradientTape context
            optimizer.appply_gradients(zip(gradients, model.variables))
            losses.append(loss.np())
    print(np.mean(losses))