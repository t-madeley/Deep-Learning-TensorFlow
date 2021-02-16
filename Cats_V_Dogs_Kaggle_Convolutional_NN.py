import wget
import os
import zipfile

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import random
import shutil
from shutil import copyfile

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))



data_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"

wget.download(data_url)
local_zip = "kagglecatsanddogs_3367a.zip"

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall()

zip_ref.close()

base_dir = 'PetImages'

os.mkdir('cats-v-dogs')
os.mkdir('cats-v-dogs/training')
os.mkdir('cats-v-dogs/testing')
os.mkdir('cats-v-dogs/training/cats')
os.mkdir('cats-v-dogs/training/dogs')
os.mkdir('cats-v-dogs/testing/cats')
os.mkdir('cats-v-dogs/testing/dogs')


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dataset = []

    # iterates over each file in the source directory (/tmp/PetImages/Cat/ or /tmp/PetImages/Dog/)
    for file in os.listdir(SOURCE):
        # creates the path for each file (/tmp/PetImages/Cat/"filenamehere")
        data = SOURCE + file
        # checks filesize not zero
        if (os.path.getsize(data) > 0):
            # appends file to the dataset
            dataset.append(file)
        else:
            print('Skipped' + file)
            print('0 File Size')
    length_training_data = int(len(dataset) * SPLIT_SIZE)
    length_test_data = int(len(dataset) - length_training_data)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = shuffled_set[0:length_training_data]
    test_set = shuffled_set[-length_test_data:]

    for filename in train_set:
        temp_train_data = SOURCE + filename
        final_train_data = TRAINING + filename
        copyfile(temp_train_data, final_train_data)

    for filename in test_set:
        temp_test_data = SOURCE + filename
        final_test_data = TESTING + filename
        copyfile(temp_test_data, final_test_data)


CAT_SOURCE_DIR = "PetImages/Cat/"
TRAINING_CATS_DIR = "cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "PetImages/Dog/"
TRAINING_DOGS_DIR = "cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "cats-v-dogs/testing/dogs/"
TRAINING_DIR = ('cats-v-dogs/training/')
VALIDATION_DIR = ('cats-v-dogs/testing/')

split_size = .9
#split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
#split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


train_cat_fnames = os.listdir(TRAINING_CATS_DIR)
train_dog_fnames = os.listdir(TRAINING_DOGS_DIR)

print(train_dog_fnames[:10])
print(train_cat_fnames[:10])


print('total training cat images :', len(os.listdir(TRAINING_CATS_DIR)))
print('total training dog images :', len(os.listdir(TRAINING_DOGS_DIR)))

print('total validation cat images :', len(os.listdir(TESTING_CATS_DIR)))
print('total validation dog images :', len(os.listdir(TESTING_DOGS_DIR)))


#Building a convolutional model:

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip=True,
                                   fill_mode= 'nearest')

test_datagen = ImageDataGenerator(rescale= 1.0/255)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=50, class_mode='binary', target_size=(150,150))

validation_generator = test_datagen.flow_from_directory(VALIDATION_DIR, batch_size=50, class_mode='binary', target_size=(150,150))

history = model.fit(train_generator, validation_data=validation_generator, epochs=50, validation_steps=50, verbose = 2)

acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )

tensorflow.config.gpu.set_per_process_memory_fraction(0.9)
tensorflow.config.gpu.set_per_process_memory_growth(True)