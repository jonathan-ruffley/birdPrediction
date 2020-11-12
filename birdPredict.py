import tensorflow as tf
from tensorflow.keras import layers
#import matplotlib.pyplot as plt
import numpy as np
import random

import PIL
import PIL.Image

import os
import pathlib

#load the IMAGES
dataDirectory = './newBirds'

dataDirectory = pathlib.Path(dataDirectory)
imageCount = len(list(dataDirectory.glob('*/*.jpg')))
print('Image count: {0}\n'.format(imageCount))

#test display an image
# osprey = list(dataDirectory.glob('OSPREY/*'))
# ospreyImage = PIL.Image.open(str(osprey[random.randint(1,100)]))
# ospreyImage.show()

# nFlicker = list(dataDirectory.glob('NORTHERN FLICKER/*'))
# nFlickerImage = PIL.Image.open(str(nFlicker[random.randint(1,100)]))
# nFlickerImage.show()

#set parameters
batchSize = 32
height=224
width=224

trainData = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset='training',
    seed=324893,
    image_size=(height,width),
    batch_size=batchSize)

testData = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=324893,
    image_size=(height,width),
    batch_size=batchSize)

#sample additional images
#plt.figure(figsize=(10,10))
# for images, labels in trainData.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(classes[labels[i]])
#         plt.axis("off")
# plt.show()

#class names
classes = trainData.class_names
testClasses = testData.class_names

#buffer to hold the data in memory for faster performance
autotune = tf.data.experimental.AUTOTUNE
trainData = trainData.cache().shuffle(1000).prefetch(buffer_size=autotune)
testData = testData.cache().prefetch(buffer_size=autotune)

#augment the dataset with zoomed and rotated images
#use convolutional layers to maintain spatial information about the images
#use max pool layers to reduce
#flatten and then apply a dense layer to predict classes
model = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(height, width, 3)),
    #layers.experimental.preprocessing.RandomRotation(0.1), #rotation might not be as useful on this dataset because bird images are generally taken with specific orietations.
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(height, width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # layers.Conv2D(512, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    #layers.Conv2D(1024, 3, padding='same', activation='relu'),
    #layers.MaxPooling2D(),
    #dropout prevents overtraining by not allowing each node to see each datapoint
    layers.Dropout(0.65),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(classes))
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
    
epochs=10
history = model.fit(
    trainData,
    validation_data=testData,
    epochs=epochs)

predictions = np.array([])
labels =  np.array([])
for x, y in testData:
  predictions = np.concatenate([predictions, model.predict_classes(x)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

confusionMatrix = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()
#print('{0}'.format(confusionMatrix))

#need to export the confusion matrix to a useful format
