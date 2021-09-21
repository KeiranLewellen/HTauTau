# Imports basics

import numpy as np
#import ROOT as r
import math
import random
import matplotlib.pyplot as plt
import h5py
import tensorflow.keras.backend as K
import tensorflow as tf
import scipy as sc
from scipy.optimize import curve_fit
import pandas as pd
import json

# Imports neural net tools

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Opens files and reads data

print("Extracting")

fOne = h5py.File("training_data/comb_distcut1_5_flat_hadel_TTbar_WJets,ohe,event.z", 'r')
totalData = fOne["deepDoubleTau"][:]
print(totalData.shape)
#(1338472, 381)

# Sets controllable values

particlesConsidered = 30
entriesPerParticle = 22

svConsidered = 5
entriesPerSV = 13

eventDataLength = 32
eventStart = 20
eventLength = 11

decayTypeColumn = -1

trainingDataLength = int(len(totalData)*0.8)

validationDataLength = int(len(totalData)*0.1)

numberOfEpochs = 100
batchSize = 1024

modelName="IN_hadel_v5p1,on_TTbar_WJets,ohe,eventOld,take_2"

# Creates Training Data

print("Preparing Data")

particleDataLength = particlesConsidered * entriesPerParticle
svDataLength = svConsidered * entriesPerSV

np.random.shuffle(totalData)

labels = totalData[:, decayTypeColumn:]

particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]
svData = totalData[:, particleDataLength + eventDataLength:particleDataLength + svDataLength + eventDataLength]
eventData = totalData[:, eventStart:eventStart + eventLength]

particleTrainingData = np.transpose(
    particleData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerParticle, particlesConsidered),
    axes=(0, 2, 1))
svTrainingData = np.transpose(svData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerSV, svConsidered),
                              axes=(0, 2, 1))
eventTrainingData = eventData[0:trainingDataLength]

trainingLabels = np.array(labels[0:trainingDataLength])

particleValidationData = np.transpose(
    particleData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength,
                                                                                         entriesPerParticle,
                                                                                         particlesConsidered),
    axes=(0, 2, 1))
svValidationData = np.transpose(
    svData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength, entriesPerSV,
                                                                                   svConsidered), axes=(0, 2, 1))
eventValidationData = eventData[trainingDataLength:trainingDataLength + validationDataLength]

validationLabels = np.array(labels[trainingDataLength:trainingDataLength + validationDataLength])

particleTestData = np.transpose(particleData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerParticle, particlesConsidered),
                                axes=(0, 2, 1))
svTestData = np.transpose(svData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerSV, svConsidered), axes=(0, 2, 1))
eventTestData = eventData[trainingDataLength + validationDataLength:]

testLabels = np.array(labels[trainingDataLength + validationDataLength:])

# Defines the interaction matrices

# Defines the recieving matrix for particles
RR = []
for i in range(particlesConsidered):
    row = []
    for j in range(particlesConsidered * (particlesConsidered - 1)):
        if j in range(i * (particlesConsidered - 1), (i + 1) * (particlesConsidered - 1)):
            row.append(1.0)
        else:
            row.append(0.0)
    RR.append(row)
RR = np.array(RR)
RR = np.float32(RR)
RRT = np.transpose(RR)

# Defines the sending matrix for particles
RST = []
for i in range(particlesConsidered):
    for j in range(particlesConsidered):
        row = []
        for k in range(particlesConsidered):
            if k == j:
                row.append(1.0)
            else:
                row.append(0.0)
        RST.append(row)
rowsToRemove = []
for i in range(particlesConsidered):
    rowsToRemove.append(i * (particlesConsidered + 1))
RST = np.array(RST)
RST = np.float32(RST)
RST = np.delete(RST, rowsToRemove, 0)
RS = np.transpose(RST)

# Defines the recieving matrix for the bipartite particle and secondary vertex graph
RK = []
for i in range(particlesConsidered):
    row = []
    for j in range(particlesConsidered * svConsidered):
        if j in range(i * svConsidered, (i + 1) * svConsidered):
            row.append(1.0)
        else:
            row.append(0.0)
    RK.append(row)
RK = np.array(RK)
RK = np.float32(RK)
RKT = np.transpose(RK)

# Defines the sending matrix for the bipartite particle and secondary vertex graph
RV = []
for i in range(svConsidered):
    row = []
    for j in range(particlesConsidered * svConsidered):
        if j % svConsidered == i:
            row.append(1.0)
        else:
            row.append(0.0)
    RV.append(row)
RV = np.array(RV)
RV = np.float32(RV)
RVT = np.transpose(RV)

# Creates and trains the neural net

# Particle data interaction NN
inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")

XdotRR = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RR, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRR")(inputParticle)
XdotRS = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RS, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRS")(inputParticle)
Bpp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bpp")([XdotRR, XdotRS])

convOneParticle = Conv1D(80, kernel_size=1, activation="relu", name="convOneParticle")(Bpp)
convTwoParticle = Conv1D(50, kernel_size=1, activation="relu", name="convTwoParticle")(convOneParticle)
convThreeParticle = Conv1D(30, kernel_size=1, activation="relu", name="convThreeParticle")(convTwoParticle)

Epp = BatchNormalization(momentum=0.6, name="Epp")(convThreeParticle)

# Secondary vertex data interaction NN
inputSV = Input(shape=(svConsidered, entriesPerSV), name="inputSV")

XdotRK = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RK, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRK")(inputParticle)
YdotRV = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RV, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="YdotRV")(inputSV)
Bvp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bvp")([XdotRK, YdotRV])

convOneSV = Conv1D(80, kernel_size=1, activation="relu", name="convOneSV")(Bvp)
convTwoSV = Conv1D(50, kernel_size=1, activation="relu", name="convTwoSV")(convOneSV)
convThreeSV = Conv1D(30, kernel_size=1, activation="relu", name="convThreeSV")(convTwoSV)

Evp = BatchNormalization(momentum=0.6, name="Evp")(convThreeSV)

# Combined prediction NN
EppBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RRT, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="EppBar")(Epp)
EvpBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RKT, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="EvpBar")(Evp)
C = Lambda(lambda listOfTensors: tf.concat((listOfTensors[0], listOfTensors[1], listOfTensors[2]), axis=2), name="C")(
    [inputParticle, EppBar, EvpBar])

convPredictOne = Conv1D(80, kernel_size=1, activation="relu", name="convPredictOne")(C)
convPredictTwo = Conv1D(50, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)

O = Conv1D(24, kernel_size=1, activation="relu", name="O")(convPredictTwo)

# Event data NN
inputEvent = Input(shape=(eventLength,), name="inputEvent")

denseEventOne = Dense(80, activation="relu", name="denseEventOne")(inputEvent)
normEventOne = BatchNormalization(momentum=0.6, name="normEventOne")(denseEventOne)
denseEventTwo = Dense(50, activation="relu", name="denseEventTwo")(normEventOne)
denseEventThree = Dense(30, activation="relu", name="denseEventThree")(denseEventTwo)
normEventTwo = BatchNormalization(momentum=0.6, name="normEventTwo")(denseEventThree)

# Calculate output
OBar = Lambda(lambda tensor: tf.math.reduce_sum(tensor, axis=1), name="OBar")(O)

combined = Concatenate(axis=1,name="combined")([OBar,normEventTwo])

denseEndOne = Dense(60, activation="relu", name="denseEndOne")(combined)
normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
denseEndTwo = Dense(30, activation="relu", name="denseEndTwo")(normEndOne)
denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
output = Dense(1, activation="sigmoid", name="output")(denseEndThree)

print("Compiling")

model = Model(inputs=[inputParticle, inputSV, inputEvent], outputs=[output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print('Calculating')

modelCallbacks = [EarlyStopping(patience=10),
                  ModelCheckpoint(filepath="./data/"+modelName+".h5", save_weights_only=True,
                                  save_best_only=True)]

history = model.fit([particleTrainingData, svTrainingData, eventTrainingData], trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                    callbacks=modelCallbacks,
                    validation_data=([particleValidationData, svValidationData, eventValidationData], validationLabels))

with open("./data/"+modelName+",history.json", "w") as f:
    json.dump(history.history,f)

print("Loading weights")

model.load_weights("./data/"+modelName+".h5")

model.save("./data/"+modelName+",model")

print("Predicting")

predictions = model.predict([particleTestData, svTestData, eventTestData])
