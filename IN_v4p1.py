# Imports basics

import numpy as np
import ROOT as r
import math
import random
import matplotlib.pyplot as plt
import h5py
import keras.backend as K
import tensorflow as tf
import scipy as sc
from scipy.optimize import curve_fit
import pandas as pd

# Imports neural net tools

from tensorflow import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model

# Opens files and reads data

print("Extracting")

fOne = h5py.File("../data/hadhad/comb_distcut2_flat_hadhad_QCD.z", 'r')

totalData = fOne.get("deepDoubleTau").value
print(totalData.shape)
#(1338472, 381)

# Sets controllable values

particlesConsidered = 30
entriesPerParticle = 10

svConsidered = 5
entriesPerSV = 13

eventDataLength = 15

decayTypeColumn = 380

trainingDataLength = 1070000

validationDataLength = 130000

numberOfEpochs = 100
batchSize = 1024

modelName="IN_hadhad_v4p1,on_QCD,fillFactor=2,take2"

# Creates Training Data

print("Preparing Data")

particleDataLength = particlesConsidered * entriesPerParticle
svDataLength = svConsidered * entriesPerSV

np.random.shuffle(totalData)

labels = totalData[:, decayTypeColumn:]

particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]
svData = totalData[:, particleDataLength + eventDataLength:particleDataLength + svDataLength + eventDataLength]

particleTrainingData = np.transpose(
    particleData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerParticle, particlesConsidered),
    axes=(0, 2, 1))
svTrainingData = np.transpose(svData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerSV, svConsidered),
                              axes=(0, 2, 1))
trainingLabels = np.array(labels[0:trainingDataLength])

particleValidationData = np.transpose(
    particleData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength,
                                                                                         entriesPerParticle,
                                                                                         particlesConsidered),
    axes=(0, 2, 1))
svValidationData = np.transpose(
    svData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength, entriesPerSV,
                                                                                   svConsidered), axes=(0, 2, 1))
validationLabels = np.array(labels[trainingDataLength:trainingDataLength + validationDataLength])

particleTestData = np.transpose(particleData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerParticle, particlesConsidered),
                                axes=(0, 2, 1))
svTestData = np.transpose(svData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerSV, svConsidered), axes=(0, 2, 1))
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

# Creates all one vector

allOneVector = []
for i in range(particlesConsidered):
    allOneVector.append(1.0)
allOneVector = np.array(allOneVector)

# Saves the jet data used for a specific training instance

def save_jet_data(fileName):
    h5file = h5py.File("./data/"+fileName, "w")
    h5file.create_dataset("particleValidationData", data=particleValidationData, compression="lzf")
    h5file.create_dataset("svValidationData", data=svValidationData, compression="lzf")
    h5file.create_dataset("validationLabels", data=validationLabels, compression="lzf")
    h5file.create_dataset("particleTestData", data=particleTestData, compression="lzf")
    h5file.create_dataset("svTestData", data=svTestData, compression="lzf")
    h5file.create_dataset("testLabels", data=testLabels, compression="lzf")
    h5file.create_dataset("totalDataInfo", data=totalData[:, 0:15], compression="lzf")
    h5file.close()
    del h5file

print("Saving data")

save_jet_data(modelName+",validationData.h5")

# Opening previous data
'''
fTwo=h5py.File("./data/"+modelName+",validationData.h5",'r')
particleValidationData=fTwo.get("particleValidationData").value
svValidationData=fTwo.get("svValidationData").value
validationLabels=fTwo.get("validationLabels").value
particleTestData=fTwo.get("particleTestData").value
svTestData=fTwo.get("svTestData").value
testLabels=fTwo.get("testLabels").value
totalData=fTwo.get("totalDataInfo").value
'''

# Creates and trains the neural net

# Particle data interaction NN
inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")

XdotRR = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RR, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRR")(inputParticle)
XdotRS = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RS, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRS")(inputParticle)
Bpp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bpp")([XdotRR, XdotRS])

convOneParticle = Conv1D(60, kernel_size=1, activation="relu", name="convOneParticle")(Bpp)
convTwoParticle = Conv1D(30, kernel_size=1, activation="relu", name="convTwoParticle")(convOneParticle)
convThreeParticle = Conv1D(20, kernel_size=1, activation="relu", name="convThreeParticle")(convTwoParticle)

Epp = BatchNormalization(momentum=0.6, name="Epp")(convThreeParticle)

# Secondary vertex data interaction NN
inputSV = Input(shape=(svConsidered, entriesPerSV), name="inputSV")

XdotRK = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RK, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRK")(inputParticle)
YdotRV = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RV, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="YdotRV")(inputSV)
Bvp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bvp")([XdotRK, YdotRV])

convOneSV = Conv1D(60, kernel_size=1, activation="relu", name="convOneSV")(Bvp)
convTwoSV = Conv1D(30, kernel_size=1, activation="relu", name="convTwoSV")(convOneSV)
convThreeSV = Conv1D(20, kernel_size=1, activation="relu", name="convThreeSV")(convTwoSV)

Evp = BatchNormalization(momentum=0.6, name="Evp")(convThreeSV)

# Combined prediction NN
EppBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RRT, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="EppBar")(Epp)
EvpBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RKT, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="EvpBar")(Evp)
C = Lambda(lambda listOfTensors: tf.concat((listOfTensors[0], listOfTensors[1], listOfTensors[2]), axis=2), name="C")(
    [inputParticle, EppBar, EvpBar])

convPredictOne = Conv1D(60, kernel_size=1, activation="relu", name="convPredictOne")(C)
convPredictTwo = Conv1D(30, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)

O = Conv1D(24, kernel_size=1, activation="relu", name="O")(convPredictTwo)

# Calculate output
OBar = Lambda(lambda tensor: K.sum(tensor, axis=1), name="OBar")(O)

denseEndOne = Dense(50, activation="relu", name="denseEndOne")(OBar)
normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
denseEndTwo = Dense(20, activation="relu", name="denseEndTwo")(normEndOne)
denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
output = Dense(1, activation="sigmoid", name="output")(denseEndThree)

print("Compiling")

model = Model(inputs=[inputParticle, inputSV], outputs=[output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print('Calculating')

modelCallbacks = [EarlyStopping(patience=10),
                  ModelCheckpoint(filepath="./data/"+modelName+".h5", save_weights_only=True,
                                  save_best_only=True)]

history = model.fit([particleTrainingData, svTrainingData], trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                    callbacks=modelCallbacks,
                    validation_data=([particleValidationData, svValidationData], validationLabels))

h5History = h5py.File("./data/"+modelName+",history.h5", "w")
h5History.create_dataset("history", data=history.history, compression="lzf")

print("Loading weights")

model.load_weights("./data/"+modelName+".h5")

model.save("./data/"+modelName+",model.h5")

'''
model=load_model("./data/"+modelName+".h5",custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})
'''
print("Predicting")

predictions = model.predict([particleTestData, svTestData])

print("Predicted")
