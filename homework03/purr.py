import tensorflow as tf

class Purr():
    def __init__(self, name = "purr"):

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

        self.model = tf.keras.Model(inputs = inputs, outputs = outputs, name = name)

    def __call__(self, x):
        self.call(x)

    
    def call(self, x):
        self.model(x)

    def set_metrics(self, loss_metric, accuracy_metric):
        self.loss_metric = loss_metric
        self.accuracy_metric = accuracy_metric    

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_state()


    @tf.function
    def train(self, data, loss_function, optimiser):

        for x, target in data:
    
            with tf.GradientTape() as tape:
                pred = self.model(x)
                loss = loss_function(target, pred)

            self.loss_metric.update_states(loss)
            self.accuracy_metric.update_states(target, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))
        


    @tf.function
    def test_step(self, data, loss_function):

        for x, target in data:

            pred = self.model(x)
            loss = loss_function(target, pred)
    
            self.loss_metric.update_states(loss)
            self.accuracy_metric.update_states(target, pred)



