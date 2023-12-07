import tensorflow as tf

class Taken_Model(tf.keras.Model):
    def __init__(self, name = "taken_model"):
        
        super().__init__()

        inputs = tf.keras.Input((32, 32, 3), dtype = tf.float32)
        x = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), activation = tf.nn.relu)(x)
        x = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), activation = tf.nn.relu)(x)

        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = tf.nn.relu)(x)
        x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = tf.nn.relu)(x)

        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.GlobalMaxPooling2D()(x)

        x = tf.keras.layers.Dense(units = 32, activation = tf.nn.relu)(x)

        outputs = tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax)(x)

        self.model = tf.keras.Model(inputs, outputs, name = name)


    def __call__(self, x):
        self.call(x)

    def call(self, x):
        self.model(x)


    def set_metrics(self, loss_metric, accuracy_metric):
        self.loss_metric = loss_metric
        self.accuracy_metric = accuracy_metric 

    def get_metrics(self):
        return self.loss_metric, self.accuracy_metric
  
    def reset_metrics(self): 
        self.loss_metric.reset_states()
        self.accuracy_metric.reset_states()

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function


    @tf.function
    def train_step(self, data, optimiser):

        for x, target in data:
    
            with tf.GradientTape() as tape:
                pred = self.model(x)
                loss = self.loss_function(target, pred)

            self.loss_metric.update_states(loss)
            self.accuracy_metric.update_states(target, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))
        


    @tf.function
    def test_step(self, data):

        for x, target in data:

            pred = self.model(x)
            loss = self.loss_function(target, pred)
    
            self.loss_metric.update_states(loss)
            self.accuracy_metric.update_states(target, pred)
