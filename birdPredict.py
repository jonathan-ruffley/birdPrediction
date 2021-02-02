#settings
import logging
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import warnings
warnings.filterwarnings('ignore')

import tensorflow_addons as tfa
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import random
import PIL
import PIL.Image
import pathlib
import subprocess
import math as m

#settings
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')
# sys.stderr = stderr

#load the IMAGES
dataDirectory = './newBirds'
dataDirectory = pathlib.Path(dataDirectory)
imageCountProcess = subprocess.Popen('find -type f -name "*jpg" | wc -l', stdout=subprocess.PIPE, shell=True)
imageCount = int(imageCountProcess.communicate()[0].strip())
#print('Image count: {0}'.format(imageCount))

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
trainTestSplit = 0.2
randomSeed = random.randint(0,1000000000)

trainData = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
    labels='inferred',
    label_mode='categorical',
    validation_split=trainTestSplit,
    subset='training',
    seed=randomSeed,
    image_size=(height,width),
    batch_size=batchSize)

testData = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
    labels='inferred',
    label_mode='categorical',
    validation_split=trainTestSplit,
    subset='validation',
    seed=randomSeed,
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

#check classes
# print(trainData.class_names)
# print(len(classes))

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

initialLearningRate = 0.0009 #try decreasing this value because the loss increases for the first part of each epoch
finalLearningRate = 0.0001
#maximalLearningRate = 0.001
epochs = 25

#linear decay
learningRate = tf.keras.optimizers.schedules.InverseTimeDecay(
                      initial_learning_rate = initialLearningRate, 
                      decay_steps = 1,
                      decay_rate = 0.000001, #(imageCount * (1 - trainTestSplit)) / ((initialLearningRate - finalLearningRate) * batchSize),
                      staircase = False
)

#cyclical
#learningRate = tfa.optimizers.CyclicalLearningRate( \
                      #initial_learning_rate = initialLearningRate, \
                      #maximal_learning_rate = maximalLearningRate, \
                      #step_size = (imageCount * (1 - trainTestSplit)) / ((maximalLearningRate - initialLearningRate) * batchSize),
                      #scale_fn = lambda x: 1.0,
                      #scale_mode = 'iterations'
#)
opt = tf.keras.optimizers.Adam(learning_rate=learningRate)
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
    
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

#plot the confusion matrix
sb.heatmap(confusionMatrix, cmap='vlag') #, xticklabels=labels, yticklabels=labels)
#plt.yticks(rotation=0, fontsize=6)
#plt.xticks(fontsize=6)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.tight_layout()
plt.show()
plt.savefig('ConfusionMatrix.png')

#analysis of model
trace = np.trace(confusionMatrix)
matrixSum = np.sum(confusionMatrix)
diagonal = np.diag(confusionMatrix)

#properties
accuracy = trace / matrixSum
falsePositives = np.sum(confusionMatrix, axis=0) - diagonal
falseNegatives = np.sum(confusionMatrix, axis=1) - diagonal
truePositives = diagonal

trueNegatives = []
for j in range(len(classes)):
  tempMatrix = np.delete(confusionMatrix, j, 0)
  tempMatrix = np.delete(tempMatrix, j, 1)
  trueNegatives.append(sum(sum(tempMatrix)))

#descriptive statistics 
recall = truePositives / (truePositives + falseNegatives) #how many of the category are recognized as being in that category
precision = truePositives / (truePositives + falsePositives) #how many of those picked for a cateogry are actually that category
specificity = trueNegatives / (trueNegatives + falsePositives) #how many of those not in a catgory are correctly noted as not in the category
fScore = (2 * precision * recall) / (precision + recall)

if len(trueNegatives) != len(truePositives) or len(trueNegatives) != len(falseNegatives) or len(trueNegatives) != len(falsePositives):
  print('Error: inequivalent vector lengths')

#test for incorrect calculations. Sum for all components of each class should be total number of images IN THE TEST SET
print('Images: {0}'.format(imageCount))
testImages = m.floor(imageCount*trainTestSplit)

for k in range(len(classes)):
  images = falsePositives[k] + falseNegatives[k] + truePositives[k] + trueNegatives[k]
  if images != testImages:
    print('Error in counts: {0} {1}'.format(images, testImages))

#print a matrix with properties for each label as columns. Could add specificity here but it is consistently > 0.98 for every class, does not yield useful info.
tempMatrix = np.transpose(np.array([classes, falsePositives, falseNegatives, truePositives, recall, precision, fScore]))
PM = tempMatrix[tempMatrix[:,4].argsort()]
header = ['Class', 'FP', 'FN', 'TP', 'Recall', 'Precision', 'F-1 Score']
print('\n\n{0:27}{1:>4}{2:>4}{3:>4}{4:>8}{5:>12}{6:>12}'.format(header[0], header[1], header[2], header[3], header[4], header[5], header[6]))
for i in range(len(classes)):
  print('{0:27}{1:>4}{2:>4}{3:>4}{4:>8.4}{5:>12.4}{6:>12.4}'.format(PM[i][0], PM[i][1], PM[i][2], PM[i][3], PM[i][4], PM[i][5], PM[i][6]))

#next steps: add a learning rate schedule to reduce number of epochs needed. Add a loop and run 3 times, save the performance matrix each time and rank the classes from worst to best performance to evaluate opportunities for data set augmentation.
