import tensorflow as tf

class FedSmoothieModel(tf.keras.Model):

    def __init__(self, model):
        super(FedSmoothieModel, self).__init__()
        self.local_model = model

    def compile(self, optimizer, loss, metrics, beta=.2):
        super(FedSmoothieModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.beta = beta

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            local_output = self.local_model(x, training=True)  # Forward pass
            y_pred = local_output
            y_one_hot = tf.one_hot(tf.squeeze(y), depth=local_output.shape[1])

            y_evenly_distrib = tf.fill(
                (tf.shape(y)[0], tf.shape(y)[1]),
                1.0/tf.cast(tf.shape(y_pred)[1], tf.float32)
            )
            y_smoothed = y_evenly_distrib * self.beta + (1 - self.beta) * y_one_hot

            loss = self.compiled_loss(y_smoothed, y_pred)

        # Compute gradients
        trainable_vars = self.local_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

        result = {m.name: m.result() for m in self.metrics}
        # result.update({"loss_feature": loss_features})
        return result

    def test_step(self, data):
        x, y = data
        y_pred = self.local_model(x, training=False)  # Forward pass
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        return self.local_model.get_weights()

    def get_global_weights(self):
        return self.global_model.get_weights()
