import tensorflow as tf

class Purr():
    def __init__(self, name = "purr"):
        
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


        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        outputs = tf.keras.layers.Dense(10, activation = tf.nn.softmax)(x)

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



