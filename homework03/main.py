import tensorflow as tf
import func

if __name__ == "__main__":
    LEARNING_RATE0 = 0.0001
    LEARNING_RATE1 = 0.03

    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    
    optimizer0 = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE0)
    optimizer1 = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE1)
    
    loss_f0 = tf.keras.losses.CategoricalCrossentropy()
    loss_m = tf.keras.metrics.Mean(name = 'loss')
    acc_m = tf.keras.metrics.Accuracy(name = 'acc')

    model0 = func.this_cnn()
    model1 = func.that_cnn()

    train_ds, test_ds = func.load_and_prep_cifar(BATCH_SIZE)



    func.train_loop(model0, optimizer0, loss_f0, train_ds, test_ds, NUM_EPOCHS, loss_m, acc_m)
    func.train_loop(model1, optimizer0, loss_f0, train_ds, test_ds, NUM_EPOCHS, loss_m, acc_m)


