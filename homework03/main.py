import tensorflow as tf
import func
import purr
import meow

if __name__ == "__main__":
    LEARNING_RATE0 = 0.0001
    LEARNING_RATE1 = 0.03

    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    
    optimiser0 = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE0)
    optimiser1 = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE1)
    
    loss_f = tf.keras.losses.CategoricalCrossentropy()
    loss_m = tf.keras.metrics.Mean(name = 'loss')
    acc_m = tf.keras.metrics.CategoricalAccuracy(name = 'acc')

    model0 = purr.Purr()
    model1 = meow.Meow()

    model0.set_metrics(loss_m, acc_m)
    model1.set_metrics(loss_m, acc_m)

    #model0.set_loss_function(loss_f)
    #model1.set_loss_function(loss_f)

    metrics = []

    train_ds, test_ds = func.load_and_prep_cifar(BATCH_SIZE)

    metrics.append(func.train_loop(model0, train_ds, test_ds, loss_f, optimiser0, NUM_EPOCHS))

    #optimiser0 = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE0)

    #metrics.append(func.train_loop(model1, train_ds, test_ds, optimiser0, NUM_EPOCHS))


    print(metrics)



