import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data(filename):
    with open(filename) as training_file:
        csv_read = csv.reader(training_file, delimiter=',')
        first_line = True
        temp_images = []
        temp_labels = []

        for row in csv_read:
            if first_line:
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_array = np.array_split(image_data,28)
                temp_images.append(image_array)

        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
    return images, labels


path_sign_mnist_train = "sign_mnist/sign_mnist_train.csv"
path_sign_mnist_test = "sign_mnist/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
test_images, test_labels = get_data(path_sign_mnist_test)


print(training_images.shape)
print(test_images.shape)
print(training_labels.shape)
print(test_labels.shape)


training_images = np.expand_dims(training_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_datagen = ImageDataGenerator( rescale = 1/255.0,
                                    rotation_range = 40,
                                    width_shift_range=.2,
                                    height_shift_range = .2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale= 1./255)

print(training_images.shape)
print(test_images.shape)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])

history = model.fit(train_datagen.flow(training_images, training_labels, batch_size = 32),
                    epochs = 50, steps_per_epoch=len(training_images)/32,
                    validation_data=validation_datagen.flow(test_images, test_labels, batch_size=32),
                    validation_steps=len(test_images)/32)
model.evaluate(test_images, test_labels, verbose=1)

