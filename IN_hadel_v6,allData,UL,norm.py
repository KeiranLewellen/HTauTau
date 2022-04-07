# Imports basics

import numpy as np
# import ROOT as r
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
import os

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

# Sets controllable values

# Data set
data_set = "comb_distcut,0_75,flat_hadel_TTbar_WJets,allData,sigShaped,metCut40,UL,multiclass.z"
base = "/nobackup/users/keiran/training_data/"
tag = "hadel_UL/Jan17/"

# Data
particlesConsidered = 30
entriesPerParticle = 23

svConsidered = 5
entriesPerSV = 13

elecConsidered = 2
entriesPerElec = 20

muonConsidered = 2
entriesPerMuon = 16

tauConsidered = 3
entriesPerTau = 14

eventDataLength = 29
eventStart = 19
eventLength = 9

decayTypeColumn = -3

# Model

multiclass = False

numberOfEpochs = 100
batchSize = 1024

modelName = "IN_hadel_v6,on_TTbar_WJets,ohe,allData,UL,metCut40,sigShaped,norm,take_5"

# Plots

pTBinRange = [200., 800.]
massBinRange = [10., 400.]

desiredFPRs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

# Opens files and reads data

dir_out = "/home/keiran/Models/data/" + modelName + "/"
os.system("mkdir " + dir_out)

dir_base = base + tag

print("Extracting")

fOne = h5py.File(dir_base + data_set, 'r')
totalData = fOne["deepDoubleTau"][:].astype("float64")
print(totalData.shape)

trainingDataLength = int(len(totalData) * 0.8)

validationDataLength = int(len(totalData) * 0.1)

# Creates Training Data

print("Preparing Data")

particleDataLength = particlesConsidered * entriesPerParticle
svDataLength = svConsidered * entriesPerSV
elecDataLength = elecConsidered * entriesPerElec
muonDataLength = muonConsidered * entriesPerMuon
tauDataLength = tauConsidered * entriesPerTau

np.random.shuffle(totalData)

if multiclass:
    labels = totalData[:, decayTypeColumn:]
else:
    labels = totalData[:, decayTypeColumn]

particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]
svData = totalData[:, particleDataLength + eventDataLength:particleDataLength + svDataLength + eventDataLength]
tauData = totalData[:,
          particleDataLength + eventDataLength + svDataLength:particleDataLength + svDataLength + tauDataLength + eventDataLength]
elecData = totalData[:,
           particleDataLength + eventDataLength + svDataLength + tauDataLength:particleDataLength + svDataLength + tauDataLength + elecDataLength + eventDataLength]
muonData = totalData[:,
           particleDataLength + eventDataLength + svDataLength + tauDataLength + elecDataLength:particleDataLength + svDataLength + tauDataLength + elecDataLength + muonDataLength + eventDataLength]
eventData = totalData[:, eventStart:eventStart + eventLength]


# Training

particleTrainingData = np.transpose(
    particleData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerParticle, particlesConsidered),
    axes=(0, 2, 1))
svTrainingData = np.transpose(svData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerSV, svConsidered),
                              axes=(0, 2, 1))
elecTrainingData = np.transpose(
    elecData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerElec, elecConsidered),
    axes=(0, 2, 1))
muonTrainingData = np.transpose(
    muonData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerMuon, muonConsidered),
    axes=(0, 2, 1))
tauTrainingData = np.transpose(
    tauData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerTau, tauConsidered),
    axes=(0, 2, 1))
eventTrainingData = eventData[0:trainingDataLength]

trainingLabels = np.array(labels[0:trainingDataLength])


# Validation

particleValidationData = np.transpose(
    particleData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength,
                                                                                         entriesPerParticle,
                                                                                         particlesConsidered),
    axes=(0, 2, 1))
svValidationData = np.transpose(
    svData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength, entriesPerSV,
                                                                                   svConsidered), axes=(0, 2, 1))
elecValidationData = np.transpose(
    elecData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength,
                                                                                     entriesPerElec,
                                                                                     elecConsidered), axes=(0, 2, 1))
muonValidationData = np.transpose(
    muonData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength,
                                                                                     entriesPerMuon,
                                                                                     muonConsidered), axes=(0, 2, 1))
tauValidationData = np.transpose(
    tauData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength, entriesPerTau,
                                                                                    tauConsidered), axes=(0, 2, 1))
eventValidationData = eventData[trainingDataLength:trainingDataLength + validationDataLength]

validationLabels = np.array(labels[trainingDataLength:trainingDataLength + validationDataLength])

# Testing

particleTestData = np.transpose(particleData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerParticle, particlesConsidered),
    axes=(0, 2, 1))
svTestData = np.transpose(svData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerSV, svConsidered), axes=(0, 2, 1))
elecTestData = np.transpose(elecData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerElec, elecConsidered), axes=(0, 2, 1))
muonTestData = np.transpose(muonData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerMuon, muonConsidered), axes=(0, 2, 1))
tauTestData = np.transpose(tauData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerTau, tauConsidered), axes=(0, 2, 1))
eventTestData = eventData[trainingDataLength + validationDataLength:]

testLabels = np.array(labels[trainingDataLength + validationDataLength:])

# Defines the interaction matrices

# Defines the receiving matrix for particles
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


# Defines the receiving matrix for the bipartite particle and secondary vertex graph
def make_RK(object_num):
    RK = []
    for i in range(particlesConsidered):
        row = []
        for j in range(particlesConsidered * object_num):
            if j in range(i * object_num, (i + 1) * object_num):
                row.append(1.0)
            else:
                row.append(0.0)
        RK.append(row)
    RK = np.array(RK)
    RK = np.float32(RK)
    RKT = np.transpose(RK)
    return RK


RK_SV = make_RK(svConsidered)
RK_el = make_RK(elecConsidered)
RK_mu = make_RK(muonConsidered)
RK_tau = make_RK(tauConsidered)


# Defines the sending matrix for the bipartite particle and secondary vertex graph
def make_RV(object_num):
    RV = []
    for i in range(object_num):
        row = []
        for j in range(particlesConsidered * object_num):
            if j % object_num == i:
                row.append(1.0)
            else:
                row.append(0.0)
        RV.append(row)
    RV = np.array(RV)
    RV = np.float32(RV)
    return RV


RV_SV = make_RV(svConsidered)
RV_el = make_RV(elecConsidered)
RV_mu = make_RV(muonConsidered)
RV_tau = make_RV(tauConsidered)


# Creates Training Data

# Saves the jet data used for a specific training instance

def save_jet_data(fileName):
    h5file = h5py.File(fileName, "w")
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
'''
#save_jet_data(modelName+",validationData.h5")
'''
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
# Creates decay information for test data

testDecays = totalData[trainingDataLength + validationDataLength:, 4:6]

# Creates and trains the neural net

# Particle data interaction NN
inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")

inputNormPart = BatchNormalization(momentum=0.6, name="inputNormPart")(inputParticle)

XdotRR = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RR, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRR")(inputNormPart)
XdotRS = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RS, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRS")(inputNormPart)
Bpp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bpp")([XdotRR, XdotRS])

convOneParticle = Conv1D(80, kernel_size=1, activation="relu", name="convOneParticle")(Bpp)
convTwoParticle = Conv1D(50, kernel_size=1, activation="relu", name="convTwoParticle")(convOneParticle)
convThreeParticle = Conv1D(30, kernel_size=1, activation="relu", name="convThreeParticle")(convTwoParticle)

Epp = BatchNormalization(momentum=0.6, name="Epp")(convThreeParticle)

# Secondary vertex data interaction NN
inputSV = Input(shape=(svConsidered, entriesPerSV), name="inputSV")

inputNormSV = BatchNormalization(momentum=0.6, name="inputNormSV")(inputSV)

XdotRK_SV = Lambda(
    lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RK_SV, axes=[[2], [0]]),
                                perm=(0, 2, 1)), name="XdotRK_SV")(inputNormPart)
YdotRV_SV = Lambda(
    lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RV_SV, axes=[[2], [0]]),
                                perm=(0, 2, 1)), name="YdotRV_SV")(inputNormSV)
Bvp_SV = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bvp")(
    [XdotRK_SV, YdotRV_SV])

convOneSV = Conv1D(80, kernel_size=1, activation="relu", name="convOneSV")(Bvp_SV)
convTwoSV = Conv1D(50, kernel_size=1, activation="relu", name="convTwoSV")(convOneSV)
convThreeSV = Conv1D(30, kernel_size=1, activation="relu", name="convThreeSV")(convTwoSV)

Evp_SV = BatchNormalization(momentum=0.6, name="Evp_SV")(convThreeSV)

# Electron data interaction NN
inputEl = Input(shape=(elecConsidered, entriesPerElec), name="inputEl")

inputNormEl = BatchNormalization(momentum=0.6, name="inputNormEl")(inputEl)

XdotRK_el = Lambda(
    lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RK_el, axes=[[2], [0]]),
                                perm=(0, 2, 1)), name="XdotRK_elec")(inputNormPart)
YdotRV_el = Lambda(
    lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RV_el, axes=[[2], [0]]),
                                perm=(0, 2, 1)), name="YdotRV_elec")(inputNormEl)
Bvp_el = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bvp_el")(
    [XdotRK_el, YdotRV_el])

convOneEl = Conv1D(80, kernel_size=1, activation="relu", name="convOneEl")(Bvp_el)
convTwoEl = Conv1D(50, kernel_size=1, activation="relu", name="convTwoEl")(convOneEl)
convThreeEl = Conv1D(30, kernel_size=1, activation="relu", name="convThreeEl")(convTwoEl)

Evp_el = BatchNormalization(momentum=0.6, name="Evp_el")(convThreeEl)

# Tau data interaction NN
inputTau = Input(shape=(tauConsidered, entriesPerTau), name="inputTau")

inputNormTau = BatchNormalization(momentum=0.6, name="inputNormTau")(inputTau)

XdotRK_tau = Lambda(
    lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RK_tau, axes=[[2], [0]]),
                                perm=(0, 2, 1)), name="XdotRK_tau")(inputNormPart)
YdotRV_tau = Lambda(
    lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RV_tau, axes=[[2], [0]]),
                                perm=(0, 2, 1)), name="YdotRV_tau")(inputNormTau)
Bvp_tau = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bvp_tau")(
    [XdotRK_tau, YdotRV_tau])

convOneTau = Conv1D(80, kernel_size=1, activation="relu", name="convOneTau")(Bvp_tau)
convTwoTau = Conv1D(50, kernel_size=1, activation="relu", name="convTwoTau")(convOneTau)
convThreeTau = Conv1D(30, kernel_size=1, activation="relu", name="convThreeTau")(convTwoTau)

Evp_tau = BatchNormalization(momentum=0.6, name="Evp_tau")(convThreeTau)

# Combined prediction NN
EppBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RRT, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="EppBar")(Epp)
EvpBar_SV = Lambda(lambda tensor: tf.transpose(
    tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), np.transpose(RK_SV), axes=[[2], [0]]),
    perm=(0, 2, 1)), name="EvpBar_SV")(Evp_SV)
EvpBar_el = Lambda(lambda tensor: tf.transpose(
    tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), np.transpose(RK_el), axes=[[2], [0]]),
    perm=(0, 2, 1)), name="EvpBar_el")(Evp_el)
EvpBar_tau = Lambda(lambda tensor: tf.transpose(
    tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), np.transpose(RK_tau), axes=[[2], [0]]),
    perm=(0, 2, 1)), name="EvpBar_tau")(Evp_tau)
C = Lambda(lambda listOfTensors: tf.concat(
    (listOfTensors[0], listOfTensors[1], listOfTensors[2], listOfTensors[3], listOfTensors[4]), axis=2), name="C")(
    [inputParticle, EppBar, EvpBar_SV, EvpBar_el, EvpBar_tau])

convPredictOne = Conv1D(80, kernel_size=1, activation="relu", name="convPredictOne")(C)
convPredictTwo = Conv1D(50, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)

O = Conv1D(24, kernel_size=1, activation="relu", name="O")(convPredictTwo)

# Event data NN
inputEvent = Input(shape=(eventLength,), name="inputEvent")

inputNormEvent = BatchNormalization(momentum=0.6, name="inputNormEvent")(inputEvent)

denseEventOne = Dense(80, activation="relu", name="denseEventOne")(inputNormEvent)
normEventOne = BatchNormalization(momentum=0.6, name="normEventOne")(denseEventOne)
denseEventTwo = Dense(50, activation="relu", name="denseEventTwo")(normEventOne)
denseEventThree = Dense(30, activation="relu", name="denseEventThree")(denseEventTwo)
normEventTwo = BatchNormalization(momentum=0.6, name="normEventTwo")(denseEventThree)

# Calculate output
OBar = Lambda(lambda tensor: tf.math.reduce_sum(tensor, axis=1), name="OBar")(O)

combined = Concatenate(axis=1, name="combined")([OBar, normEventTwo])

denseEndOne = Dense(60, activation="relu", name="denseEndOne")(combined)
normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
denseEndTwo = Dense(30, activation="relu", name="denseEndTwo")(normEndOne)
denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
output = Dense(1, activation="sigmoid", name="output")(denseEndThree)

print("Compiling")

model = Model(inputs=[inputParticle, inputSV, inputEl, inputTau, inputEvent], outputs=[output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print('Calculating')

modelCallbacks = [EarlyStopping(patience=10),
                  ModelCheckpoint(filepath=dir_out + modelName + ".h5", save_weights_only=True,
                                  save_best_only=True)]

history = model.fit([particleTrainingData, svTrainingData, elecTrainingData, tauTrainingData, eventTrainingData],
                    trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                    callbacks=modelCallbacks,
                    validation_data=([particleValidationData, svValidationData, elecValidationData, tauValidationData,
                                      eventValidationData], validationLabels))

history_df = pd.DataFrame.from_dict(history.history)
history_df.to_csv(dir_out + modelName + ",history.json")

print("Loading weights")

model.load_weights(dir_out + modelName + ".h5")

model.save(dir_out + modelName + ",model")

print("Predicting")

predictions = model.predict([particleTestData, svTestData, elecTestData, tauTestData, eventTestData])

print("Predicted")
