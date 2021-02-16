import tensorflow as tf

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# scaling the values to between 0 and 1
training_images = training_images.reshape(60000, 28,28,1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28,28,1)
test_images = test_images / 255.0


class Mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.998:
            print("/n reached 99.8% accuracy, stopping training")
            self.model.stop_training = True


callbacks = Mycallbacks()


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation ='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit(training_images, training_labels, epochs=20, callbacks = [callbacks])

history.epoch
history.history['acc'][-1]
