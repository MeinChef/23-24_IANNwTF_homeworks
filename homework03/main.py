from imports import tf
import func
import purr
import meow
import stolen_model

if __name__ == "__main__":

    #########################
    # SAVE OPTIMISER IN MODEL - done
    #########################
    LEARNING_RATE0 = 0.01
    LEARNING_RATE1 = 0.03

    BATCH_SIZE = 128
    NUM_EPOCHS = 25
    
    # optimiser0 = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE0)
    # optimiser1 = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE1)
    # 
    # loss_f = tf.keras.losses.CategoricalCrossentropy()
    # loss_m = tf.keras.metrics.Mean(name = 'loss')
    # acc_m = tf.keras.metrics.CategoricalAccuracy(name = 'acc')

    # store the models in a dictionary, so that the names list becomes obsolete, and we can have the models numbered from 0-8
    # memory management - delete one model after training
    model0 = purr.Purr()
    model1 = meow.Meow()
    model_stolen = stolen_model.stolen_Model()

    model0.set_metrics()
    model1.set_metrics()
    model_stolen.set_metrics()

    model0.set_loss_function()
    model1.set_loss_function()
    model_stolen.set_loss_function()

    model0.set_optimiser(learning_rate = LEARNING_RATE0)
    model1.set_optimiser(learning_rate = LEARNING_RATE0)
    model_stolen.set_optimiser(learning_rate = LEARNING_RATE0)

    metrics = []
    names = ['Purr', 'Meow', 'stolen model']

    train_ds, test_ds = func.load_and_prep_cifar(BATCH_SIZE)

    metrics.append(model0.train_loop(train_ds, test_ds, NUM_EPOCHS))
    metrics.append(model1.train_loop(train_ds, test_ds, NUM_EPOCHS))
    metrics.append(model_stolen.train_loop(train_ds, test_ds, NUM_EPOCHS))

    func.visualise(metrics, names)

    print(metrics)



