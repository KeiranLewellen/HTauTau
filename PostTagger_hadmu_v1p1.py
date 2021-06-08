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
import json

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
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model

# Opens files and reads data

print("Extracting")

fOne = h5py.File("/data/t3home000/keiran/HtautauData_taus/hadmu/comb_distcut1_5_flat_hadmu_TTbar_WJets,ohe_taus.z", 'r')
totalData = fOne.get("deepDoubleTau").value
print(totalData.shape)
#(1338472, 381)

# Sets controllable values

particlesConsidered = 30
entriesPerParticle = 20

svConsidered = 5
entriesPerSV = 13

tausConsidered = 3
entriesPerTau = 18

eventDataLength = 20

decayTypeColumn = -1

#trainingDataLength = 744000
#validationDataLength = 93000

trainingDataLength = int(len(totalData)*0.8)

validationDataLength = int(len(totalData)*0.1)

numberOfEpochs = 100
batchSize = 1024

INModelName = "IN_hadmu_v4p1,on_TTbar_Wjets,fillFactor=1_5,200GeV,ohe,take_1"

modelName="PostTagger_hadmu_v1p1,on_TTbar_WJets,take_1"

# Creates Training Data

print("Preparing Data")

particleDataLength = particlesConsidered * entriesPerParticle
svDataLength = svConsidered * entriesPerSV
tauDataLength = tausConsidered * entriesPerTau

np.random.shuffle(totalData)

labels = totalData[:, decayTypeColumn:]

particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]
svData = totalData[:, particleDataLength + eventDataLength:particleDataLength + svDataLength + eventDataLength]
tauData = totalData[:, svDataLength + particleDataLength + eventDataLength:particleDataLength + svDataLength + tauDataLength + eventDataLength]

particleTrainingData = np.transpose(
    particleData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerParticle, particlesConsidered),
    axes=(0, 2, 1))
svTrainingData = np.transpose(svData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerSV, svConsidered),
                              axes=(0, 2, 1))
tauTrainingData = np.transpose(tauData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerTau, tausConsidered),
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
tauValidationData = np.transpose(
    tauData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength, entriesPerTau,
                                                                                   tausConsidered), axes=(0, 2, 1))
validationLabels = np.array(labels[trainingDataLength:trainingDataLength + validationDataLength])

particleTestData = np.transpose(particleData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerParticle, particlesConsidered),
                                axes=(0, 2, 1))
svTestData = np.transpose(svData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerSV, svConsidered), axes=(0, 2, 1))
tauTestData = np.transpose(tauData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerTau, tausConsidered), axes=(0, 2, 1))
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

# Loads the IN tagger

IN_model = load_model("./data/"+INModelName+",model.h5",custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})

print("Creating IN tagger predictions")
print("Test data")
INTestData = IN_model.predict([particleTestData, svTestData])

print("Validation data")
INValidationData = IN_model.predict([particleValidationData, svValidationData])

print("Training data")
INTrainingData = IN_model.predict([particleTrainingData, svTrainingData])

# Saves the jet data used for a specific training instance

def save_jet_data(fileName):
    h5file = h5py.File("./data/"+fileName, "w")
    h5file.create_dataset("particleValidationData", data=particleValidationData, compression="lzf")
    h5file.create_dataset("svValidationData", data=svValidationData, compression="lzf")
    h5file.create_dataset("tauValidationData", data=tauValidationData, compression="lzf")
    h5file.create_dataset("INValidationData", data=INValidationData, compression="lzf")
    h5file.create_dataset("validationLabels", data=validationLabels, compression="lzf")
    h5file.create_dataset("particleTestData", data=particleTestData, compression="lzf")
    h5file.create_dataset("svTestData", data=svTestData, compression="lzf")
    h5file.create_dataset("tauTestData", data=tauTestData, compression="lzf")
    h5file.create_dataset("INTestData", data=INTestData, compression="lzf")
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
tauValidationData=fTwo.get("tauValidationData").value
INValidationData=fTwo.get("INValidationData").value
validationLabels=fTwo.get("validationLabels").value
particleTestData=fTwo.get("particleTestData").value
svTestData=fTwo.get("svTestData").value
tauTestData=fTwo.get("tauTestData").value
INTestData=fTwo.get("INTestData").value
testLabels=fTwo.get("testLabels").value
totalData=fTwo.get("totalDataInfo").value
'''
# Creates decay information for test data

testDecays = totalData[trainingDataLength+validationDataLength:,4:6]

# Creates and trains the neural net

# Tau data convolutional NN
inputTau = Input(shape=(tausConsidered, entriesPerTau), name="inputTau")

convOneTau = Conv1D(100,kernel_size=1,activation="relu",name="convOneTau")(inputTau)
convTwoTau = Conv1D(60,kernel_size=1,activation="relu",name="convTwoTau")(convOneTau)
convThreeTau = Conv1D(30,kernel_size=1,activation="relu",name="convThreeTau")(convTwoTau)
convFourTau = Conv1D(20,kernel_size=1,activation="relu",name="convFourTau")(convThreeTau)
normOneTau = BatchNormalization(momentum=0.6,name="normOneTau")(convFourTau)
rowTau = Flatten(name="rowTau")(normOneTau)

# IN network input
inputIN = Input(shape=(1,), name="inputIN")

# Combined network
combined = Concatenate(axis=1,name="combined")([inputIN,rowTau])

denseEndOne = Dense(60, activation="relu", name="denseEndOne")(combined)
normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
denseEndTwo = Dense(30, activation="relu", name="denseEndTwo")(denseEndOne)
denseEndThree = Dense(20, activation="relu", name="denseEndThree")(denseEndTwo)
output = Dense(1, activation="sigmoid", name="output")(denseEndThree)

print("Compiling")

model = Model(inputs=[inputTau, inputIN], outputs=[output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print('Calculating')

modelCallbacks = [EarlyStopping(patience=10),
                  ModelCheckpoint(filepath="./data/"+modelName+".h5", save_weights_only=True,
                                  save_best_only=True)]

history = model.fit([tauTrainingData, INTrainingData], trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                    callbacks=modelCallbacks,
                    validation_data=([tauValidationData, INValidationData], validationLabels))

with open("./data/"+modelName+",history.json", "w") as f:
    json.dump(history.history,f)

print("Loading weights")

model.load_weights("./data/"+modelName+".h5")

model.save("./data/"+modelName+",model.h5")

'''
IN_model=load_model("./data/"+modelName+",model.h5",custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})
'''
print("Predicting")

predictions = model.predict([tauTestData, INTestData])

print("Predicted")

# Defines various measures of accuracy

def positives_count(dataLabels):
    positivesCount = 0
    for entry in dataLabels:
        positivesCount += entry
    return float(positivesCount)


def negatives_count(dataLabels):
    return float(len(dataLabels) - positives_count(dataLabels))


def create_boolean(threshold, data):
    output = []
    for element in data:
        if element > threshold:
            output.append(1)
        else:
            output.append(0)
    positives = 0
    for entry in output:
        positives += entry
    return output


def true_positives(threshold, dataPredictions, dataLabels):
    truePositivesCount = 0
    dataPredictionsBoolean = create_boolean(threshold, dataPredictions)
    for i in range(len(dataLabels)):
        if dataPredictionsBoolean[i] == 1 and dataPredictionsBoolean[i] == dataLabels[i]:
            truePositivesCount += 1
    return truePositivesCount


def false_positives(threshold, dataPredictions, dataLabels):
    falsePositivesCount = 0
    dataPredictionsBoolean = create_boolean(threshold, dataPredictions)
    for i in range(len(dataLabels)):
        if dataPredictionsBoolean[i] == 1 and not dataPredictionsBoolean[i] == dataLabels[i]:
            falsePositivesCount += 1
    return falsePositivesCount


def true_negatives(threshold, dataPredictions, dataLabels):
    trueNegativesCount = 0
    dataPredictionsBoolean = create_boolean(threshold, dataPredictions)
    for i in range(len(dataLabels)):
        if dataPredictionsBoolean[i] == 0 and dataPredictionsBoolean[i] == dataLabels[i]:
            trueNegativesCount += 1
    return trueNegativesCount


def false_negatives(threshold, dataPredictions, dataLabels):
    falseNegativesCount = 0
    dataPredictionsBoolean = create_boolean(threshold, dataPredictions)
    for i in range(len(dataLabels)):
        if dataPredictionsBoolean[i] == 1 and dataPredictionsBoolean[i] == dataLabels[i]:
            falseNegativesCount += 1
    return falseNegativesCount


def true_positive_rate(threshold, dataPredictions, dataLabels):
    positivesCount = positives_count(dataLabels)
    if positivesCount == 0:
        return 1.0
    else:
        return true_positives(threshold, dataPredictions, dataLabels) / positivesCount


def false_positive_rate(threshold, dataPredictions, dataLabels):
    negativesCount = negatives_count(dataLabels)
    if negativesCount == 0:
        return 0.0
    else:
        return false_positives(threshold, dataPredictions, dataLabels) / negativesCount


print("Showing Data")


# Creates the ROC curve for the model

def ROC_curve(samplesNumber, predictions, validationLabels):
    '''ROC_curve(int,array(1xn),array(1xn))'''
    truePositiveRateValues = []
    rate = 1
    for i in range(samplesNumber):
        truePositiveRateValues.append(rate)
        rate -= 1.0 / float(samplesNumber)

    falsePositiveRateValues = []
    thresholds = percentile_TPR(samplesNumber, predictions, validationLabels)
    for entry in thresholds:
        print(entry)
        falsePositiveRateValues.append(false_positive_rate(entry, predictions, validationLabels))

    truePositiveRateValues.append(0.0)
    falsePositiveRateValues.append(0.0)

    print(truePositiveRateValues)
    print(falsePositiveRateValues)

    print("Preparing Graph")

    plt.plot(falsePositiveRateValues, truePositiveRateValues, 'b', label='ROC curve')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.xscale('log')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


# Creates a binned true positive rate curve for the model

def true_positive_rate_curve(binNumber, binRange, threshold, predictions, validationLabels, classificationData,
                             classificationType):
    '''true_positve_rate_curve(int,[int,int],float,array(1xn),array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    histogramData = []
    histogramError = []

    for i in range(binNumber):
        predictionsInBin = []
        labelsInBin = []
        for j in range(len(classificationData)):
            if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                    i + 1) * binWidth:
                predictionsInBin.append(predictions[j])
                labelsInBin.append(validationLabels[j])
        truePositiveRate = true_positive_rate(threshold, predictionsInBin, labelsInBin)
        positivesCount = positives_count(labelsInBin)
        histogramData.append(truePositiveRate)
        if positivesCount == 0:
            histogramError.append(0)
        else:
            histogramError.append(np.sqrt(truePositiveRate * (1 - truePositiveRate) / positivesCount))

    print(histogramData)
    print(histogramError)

    xData = [minBinValue]
    for i in range(binNumber - 1):
        xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + (i + 1) * binWidth)
    xData.append(minBinValue + binNumber * binWidth)

    yData = []
    yHigh = []
    yLow = []
    for i in range(binNumber):
        yData.append(histogramData[i])
        yData.append(histogramData[i])
        yHigh.append(min(histogramData[i] + histogramError[i], 1))
        yHigh.append(min(histogramData[i] + histogramError[i], 1))
        yLow.append(max(histogramData[i] - histogramError[i], 0))
        yLow.append(max(histogramData[i] - histogramError[i], 0))

    print("Preparing Graph")

    plt.plot(xData, yData, 'b')
    plt.plot(xData, yHigh, 'r')
    plt.plot(xData, yLow, 'r')

    plt.title('True positive rate curve: ' + classificationType)
    plt.xlabel(classificationType)
    plt.ylabel('True positive rate')
    plt.show()


# Creates a binned false positive rate curve for the model

def false_positive_rate_curve(binNumber, binRange, threshold, predictions, validationLabels, classificationData,
                              classificationType):
    '''false_positve_rate_curve(int,[int,int],float,array(1xn),array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    histogramData = []
    histogramError = []

    for i in range(binNumber):
        predictionsInBin = []
        labelsInBin = []
        for j in range(len(classificationData)):
            if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                    i + 1) * binWidth:
                predictionsInBin.append(predictions[j])
                labelsInBin.append(validationLabels[j])
        falsePositiveRate = false_positive_rate(threshold, predictionsInBin, labelsInBin)
        negativesCount = negatives_count(labelsInBin)
        histogramData.append(falsePositiveRate)
        if negativesCount == 0:
            histogramError.append(0)
        else:
            histogramError.append(np.sqrt(falsePositiveRate * (1 - falsePositiveRate) / negativesCount))

    print(histogramData)
    print(histogramError)

    xData = [minBinValue]
    for i in range(binNumber - 1):
        xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + (i + 1) * binWidth)
    xData.append(minBinValue + binNumber * binWidth)

    yData = []
    yHigh = []
    yLow = []
    for i in range(binNumber):
        yData.append(histogramData[i])
        yData.append(histogramData[i])
        yHigh.append(min(histogramData[i] + histogramError[i], 1))
        yHigh.append(min(histogramData[i] + histogramError[i], 1))
        yLow.append(max(histogramData[i] - histogramError[i], 0))
        yLow.append(max(histogramData[i] - histogramError[i], 0))

    print("Preparing Graph")

    plt.plot(xData, yData, 'b')
    plt.plot(xData, yHigh, 'r')
    plt.plot(xData, yLow, 'r')

    plt.title('False positive rate curve: ' + classificationType)
    plt.xlabel(classificationType)
    plt.ylabel('False positive rate')
    plt.show()


# Creates a histogram of data for Higgs jets

def histogram_Higgs(binNumber, binRange, validationLabels, classificationData, classificationType):
    '''histogram_Higgs(int,[int,int],array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    histogramData = []

    for i in range(binNumber):
        predictionsInBin = []
        labelsInBin = []
        for j in range(len(classificationData)):
            if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                    i + 1) * binWidth:
                labelsInBin.append(validationLabels[j])
        histogramData.append(positives_count(labelsInBin))

    print(histogramData)

    xData = [minBinValue]
    for i in range(binNumber - 1):
        xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + (i + 1) * binWidth)
    xData.append(minBinValue + binNumber * binWidth)

    yData = []
    for i in range(binNumber):
        yData.append(histogramData[i])
        yData.append(histogramData[i])

    print("Preparing Graph")

    plt.plot(xData, yData, 'b')

    plt.title('Higgs jets histogram: ' + classificationType)
    plt.xlabel(classificationType)
    plt.ylabel('Number of Higgs jets')
    plt.show()


# Creates a histogram of data for non-Higgs jets

def histogram_non_Higgs(binNumber, binRange, validationLabels, classificationData, classificationType):
    '''histogram_non_Higgs(int,[int,int],array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    histogramData = []

    for i in range(binNumber):
        predictionsInBin = []
        labelsInBin = []
        for j in range(len(classificationData)):
            if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                    i + 1) * binWidth:
                labelsInBin.append(validationLabels[j])
        histogramData.append(negatives_count(labelsInBin))

    print(histogramData)

    xData = [minBinValue]
    for i in range(binNumber - 1):
        xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + (i + 1) * binWidth)
    xData.append(minBinValue + binNumber * binWidth)

    yData = []
    for i in range(binNumber):
        yData.append(histogramData[i])
        yData.append(histogramData[i])

    print("Preparing Graph")

    plt.plot(xData, yData, 'b')

    plt.title('Non-Higgs jets histogram: ' + classificationType)
    plt.xlabel(classificationType)
    plt.ylabel('Number of non-Higgs jets')
    plt.show()


# Creates a graph of binned true positve rates for many thresholds

def threshold_curve_TPR(binNumber, binRange, thresholdList, predictions, validationLabels, classificationData,
                        classificationType):
    '''threshold_curve_TPR(int,[int,int],List,array(1xn),array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    for k in range(len(thresholdList)):
        threshold = thresholdList[k]
        histogramData = []
        for i in range(binNumber):
            predictionsInBin = []
            labelsInBin = []
            for j in range(len(classificationData)):
                if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                        i + 1) * binWidth:
                    predictionsInBin.append(predictions[j])
                    labelsInBin.append(validationLabels[j])
            histogramData.append(true_positive_rate(threshold, predictionsInBin, labelsInBin))

        print(histogramData)

        xData = [minBinValue]
        for i in range(binNumber - 1):
            xData.append(minBinValue + (i + 1) * binWidth)
            xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + binNumber * binWidth)

        yData = []
        for i in range(binNumber):
            yData.append(histogramData[i])
            yData.append(histogramData[i])

        graphTypes = ['-k', "--r", ":b"]
        plt.plot(xData, yData, graphTypes[k % 3])

    print("Preparing Graph")

    plt.title('Threshold curve for true positive rate: ' + classificationType)
    plt.xlabel(classificationType)
    plt.ylabel('True positive rate')
    plt.show()


# Creates a graph of binned false positve rates for many thresholds

def threshold_curve_FPR(binNumber, binRange, thresholdList, predictions, validationLabels, classificationData,
                        classificationType):
    '''threshold_curve_FPR(int,[int,int],List,array(1xn),array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    for k in range(len(thresholdList)):
        threshold = thresholdList[k]
        histogramData = []
        for i in range(binNumber):
            predictionsInBin = []
            labelsInBin = []
            for j in range(len(classificationData)):
                if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                        i + 1) * binWidth:
                    predictionsInBin.append(predictions[j])
                    labelsInBin.append(validationLabels[j])
            histogramData.append(false_positive_rate(threshold, predictionsInBin, labelsInBin))

        print(histogramData)

        xData = [minBinValue]
        for i in range(binNumber - 1):
            xData.append(minBinValue + (i + 1) * binWidth)
            xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + binNumber * binWidth)

        yData = []
        for i in range(binNumber):
            yData.append(histogramData[i])
            yData.append(histogramData[i])

        graphTypes = ['-k', "--r", ":b"]
        plt.plot(xData, yData, graphTypes[k % 3])

    print("Preparing Graph")

    plt.title('hadhad: Threshold curve for false positive rate: ' + classificationType)
    plt.xlabel(classificationType)
    plt.xlim(binRange)
    plt.ylim(0.0001, 1)
    plt.ylabel('False positive rate')
    plt.yscale('log')
    plt.show()


# Caluculates the threshold's necessary to divide up the true positive rate by percentile

def percentile_TPR(thresholdsNumber, predictions, validationLabels):
    positivesList = []
    for i in range(len(validationLabels)):
        if validationLabels[i] == 1:
            positivesList.append(predictions[i][0])
    positivesListSorted = sorted(positivesList)
    thresholdList = []
    for i in range(thresholdsNumber):
        thresholdList.append(positivesListSorted[
                                 min(int(round(i * (len(positivesListSorted) / float(thresholdsNumber)))),
                                     len(positivesListSorted) - 1)])
    return thresholdList


# Caluculates the threshold's necessary to divide up the false positive rate by percentile

def percentile_FPR(thresholdsNumber, predictions, validationLabels):
    negativesList = []
    for i in range(len(validationLabels)):
        if validationLabels[i] == 0:
            negativesList.append(predictions[i][0])
    negativesListSorted = sorted(negativesList)
    thresholdList = []
    for i in range(thresholdsNumber):
        thresholdList.append(negativesListSorted[
                                 min(int(round(i * (len(negativesListSorted) / float(thresholdsNumber)))),
                                     len(negativesListSorted) - 1)])
    return thresholdList


# Calculates the thresholds necessary to divide up the false positive rate by the specified percentiles

def percentile_FPR_given_desired(desiredFPRs, predictions, validationLabels):
    negativesList = []
    for i in range(len(validationLabels)):
        if validationLabels[i] == 0:
            negativesList.append(predictions[i][0])
    negativesListSorted = sorted(negativesList)
    thresholdList = []
    for entry in desiredFPRs:
        if len(negativesListSorted) > 0:
            thresholdList.append(negativesListSorted[min(int(round(float(len(negativesListSorted)) * (1 - entry))),
                                                         len(negativesListSorted) - 1)])
        else:
            thresholdList.append(0.0)
    return thresholdList


# Creates 2d histogram of jet pT and jet mass for Higgs jets

def pT_mass_histogram_higgs(binsNumber, binRange, jetPt, jetMass, validationLabels):
    positivesPT = []
    positivesMass = []
    for i in range(len(validationLabels)):
        if validationLabels[i] == 1:
            positivesPT.append(jetPt[i][0])
            positivesMass.append(jetMass[i][0])
    plt.hist2d(positivesPT, positivesMass, bins=binsNumber, range=binRange)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.xlabel("Jet pT")
    plt.ylabel("Jet mass")
    plt.title("2D histogram of jet pT and jet mass for Higgs jets")
    plt.show()


# Creates 2d histogram of jet pT and jet mass for non-Higgs jets

def pT_mass_histogram_non_higgs(binsNumber, binRange, jetPt, jetMass, validationLabels):
    negativesPT = []
    negativesMass = []
    for i in range(len(validationLabels)):
        if validationLabels[i] == 0:
            negativesPT.append(jetPt[i][0])
            negativesMass.append(jetMass[i][0])
    plt.hist2d(negativesPT, negativesMass, bins=binsNumber, range=binRange)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.xlabel("Jet pT")
    plt.ylabel("Jet mass")
    plt.title("2D histogram of jet pT and jet mass for non-Higgs jets")
    plt.show()


# Saves ROC curve data for the model

def save_ROC_curve(samplesNumber, predictions, validationLabels, fileName):
    '''save_ROC_curve(int,array(1xn),array(1xn),str)'''
    truePositiveRateValues = []
    rate = 1
    for i in range(samplesNumber):
        truePositiveRateValues.append(rate)
        rate -= 1.0 / float(samplesNumber)

    falsePositiveRateValues = []
    thresholds = percentile_TPR(samplesNumber, predictions, validationLabels)
    for entry in thresholds:
        #print(entry)
        falsePositiveRateValues.append(false_positive_rate(entry, predictions, validationLabels))

    truePositiveRateValues.append(0.0)
    falsePositiveRateValues.append(0.0)

    #print(truePositiveRateValues)
    #print(falsePositiveRateValues)

    truePositiveRateValues = np.array(truePositiveRateValues)
    falsePositiveRateValues = np.array(falsePositiveRateValues)

    print("Preparing Graph")

    h5file = h5py.File(fileName, "w")
    h5file.create_dataset("TPRValues", data=truePositiveRateValues, compression="lzf")
    h5file.create_dataset("FPRValues", data=falsePositiveRateValues, compression="lzf")
    h5file.close()
    del h5file


# Two functions that togehter create the  ROC curves the decay types

# Creates the individual ROC curves
def ROC_curve_for_decay(samplesNumber, predictions, validationLabels, nonHiggsPredictions, nonHiggsLabels, color,
                        label):
    '''ROC_curve_for_decay(int,array(1xn),array(1xn),array(1xn),str,str)'''
    truePositiveRateValues = []
    rate = 1
    for i in range(samplesNumber):
        truePositiveRateValues.append(rate)
        rate -= 1.0 / float(samplesNumber)

    falsePositiveRateValues = []
    thresholds = percentile_TPR(samplesNumber, predictions, validationLabels)
    for entry in thresholds:
        print(entry)
        falsePositiveRateValues.append(false_positive_rate(entry, nonHiggsPredictions, nonHiggsLabels))

    truePositiveRateValues.append(0.0)
    falsePositiveRateValues.append(0.0)

    truePositiveRateValues = np.array(truePositiveRateValues)
    falsePositiveRateValues = np.array(falsePositiveRateValues)

    plt.plot(falsePositiveRateValues, truePositiveRateValues, color, label=label)


# Creates the information to plot
def decay_ROC_curve(samplesNumber, predictions, testLabels, testDecays):
    '''decay_efficiency_curve_TPR(int,array(1xn),array(1xn),array(2xn))'''
    '''
    bothOneProng=[[0,0,0]]
    bothPiZero=[[0,0,0]]
    bothThreeProng=[[0,0,0]]
    oneProngPlusPiZero=[[0,0,0]]
    oneProngPlusThreeProng=[[0,0,0]]
    piZeroPlusThreeProng=[[0,0,0]]
    '''
    bothOneProng = []
    bothPiZero = []
    bothThreeProng = []
    oneProngPlusPiZero = []
    oneProngPlusThreeProng = []
    piZeroPlusThreeProng = []
    QCD = []

    for i in range(len(testDecays)):
        decayInfo = testDecays[i]
        if decayInfo[0] == 0 and decayInfo[1] == 0:
            bothOneProng.append([predictions[i], testLabels[i]])
        if decayInfo[0] in [1, 2] and decayInfo[1] in [1, 2]:
            bothPiZero.append([predictions[i], testLabels[i]])
        if decayInfo[0] in [10, 11] and decayInfo[1] in [10, 11]:
            bothThreeProng.append([predictions[i], testLabels[i]])
        if decayInfo[0] == 0 and decayInfo[1] in [1, 2]:
            oneProngPlusPiZero.append([predictions[i], testLabels[i]])
        if decayInfo[0] in [1, 2] and decayInfo[1] == 0:
            oneProngPlusPiZero.append([predictions[i], testLabels[i]])
        if decayInfo[0] == 0 and decayInfo[1] in [10, 11]:
            oneProngPlusThreeProng.append([predictions[i], testLabels[i]])
        if decayInfo[0] in [10, 11] and decayInfo[1] == 0:
            oneProngPlusThreeProng.append([predictions[i], testLabels[i]])
        if decayInfo[0] in [1, 2] and decayInfo[1] in [10, 11]:
            piZeroPlusThreeProng.append([predictions[i], testLabels[i]])
        if decayInfo[0] in [10, 11] and decayInfo[1] in [1, 2]:
            piZeroPlusThreeProng.append([predictions[i], testLabels[i]])
        if testLabels[i] == 0:
            QCD.append([predictions[i], testLabels[i]])

    bothOneProng = np.array(bothOneProng)
    bothPiZero = np.array(bothPiZero)
    bothThreeProng = np.array(bothThreeProng)
    oneProngPlusPiZero = np.array(oneProngPlusPiZero)
    oneProngPlusThreeProng = np.array(oneProngPlusThreeProng)
    piZeroPlusThreeProng = np.array(piZeroPlusThreeProng)
    QCD = np.array(QCD)

    print(len(bothOneProng))
    print(len(bothPiZero))
    print(len(bothThreeProng))
    print(len(oneProngPlusPiZero))
    print(len(oneProngPlusThreeProng))
    print(len(piZeroPlusThreeProng))
    print(len(QCD))

    ROC_curve_for_decay(samplesNumber, bothOneProng[:, 0], bothOneProng[:, 1], QCD[:, 0], QCD[:, 1], "-k", "1,1")
    ROC_curve_for_decay(samplesNumber, bothPiZero[:, 0], bothPiZero[:, 1], QCD[:, 0], QCD[:, 1], "-b", "1+,1+")
    ROC_curve_for_decay(samplesNumber, bothThreeProng[:, 0], bothThreeProng[:, 1], QCD[:, 0], QCD[:, 1], "-r", "3,3")
    ROC_curve_for_decay(samplesNumber, oneProngPlusPiZero[:, 0], oneProngPlusPiZero[:, 1], QCD[:, 0], QCD[:, 1], "-g",
                        "1,1+")
    ROC_curve_for_decay(samplesNumber, oneProngPlusThreeProng[:, 0], oneProngPlusThreeProng[:, 1], QCD[:, 0], QCD[:, 1],
                        "-m", "1,3")
    ROC_curve_for_decay(samplesNumber, piZeroPlusThreeProng[:, 0], piZeroPlusThreeProng[:, 1], QCD[:, 0], QCD[:, 1],
                        "-c", "1+,3")

    plt.xlim(0.000001, 1)
    plt.title('ROC curves by decay type')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.xscale('log')
    plt.show()


# Creates a graph of binned absolute positives for many thresholds

def threshold_curve_positives(binNumber, binRange, thresholdList, predictions, classificationData, classificationType):
    '''threshold_curve_positives(int,[int,int],List,array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    for k in range(len(thresholdList)):
        threshold = thresholdList[k]
        histogramData = []
        for i in range(binNumber):
            predictionsInBin = []
            for j in range(len(classificationData)):
                if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                        i + 1) * binWidth:
                    predictionsInBin.append(predictions[j])
            histogramData.append(positives_count(create_boolean(threshold, predictionsInBin)))

        print(histogramData)

        xData = [minBinValue]
        for i in range(binNumber - 1):
            xData.append(minBinValue + (i + 1) * binWidth)
            xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + binNumber * binWidth)

        yData = []
        for i in range(binNumber):
            yData.append(histogramData[i])
            yData.append(histogramData[i])

        graphTypes = ['-k', "--r", ":b"]
        plt.plot(xData, yData, graphTypes[k % 3])

    print("Preparing Graph")

    plt.title('Positives for given thresholds: ' + classificationType)
    # plt.title('Histogram of real data: '+classificationType)
    plt.xlabel(classificationType)
    plt.xlim(binRange)
    plt.ylabel('positives')
    plt.show()


# Creates a graph of binned absolute positives for Higgs only for many thresholds

def threshold_curve_positives_Higgs(binNumber, binRange, thresholdList, predictions, testLabels, classificationData,
                                    classificationType):
    '''threshold_curve_positives_Higgs(int,[int,int],List,array(1xn),array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    for k in range(len(thresholdList)):
        threshold = thresholdList[k]
        histogramData = []
        for i in range(binNumber):
            predictionsInBin = []
            for j in range(len(classificationData)):
                if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                        i + 1) * binWidth and testLabels[j] == 1:
                    predictionsInBin.append(predictions[j])
            histogramData.append(positives_count(create_boolean(threshold, predictionsInBin)))

        print(histogramData)

        xData = [minBinValue]
        for i in range(binNumber - 1):
            xData.append(minBinValue + (i + 1) * binWidth)
            xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + binNumber * binWidth)

        yData = []
        for i in range(binNumber):
            yData.append(histogramData[i])
            yData.append(histogramData[i])

        graphTypes = ['-k', "--r", ":b"]
        plt.plot(xData, yData, graphTypes[k % 3])

    print("Preparing Graph")

    plt.title('Positives on Higgs given thresholds: ' + classificationType)
    plt.xlabel(classificationType)
    plt.xlim(binRange)
    plt.ylabel('positives')
    plt.show()


# Creates a graph of binned absolute positives for QCD only for many thresholds

def threshold_curve_positives_QCD(binNumber, binRange, thresholdList, predictions, testLabels, classificationData,
                                  classificationType):
    '''threshold_curve_positives_QCD(int,[int,int],List,array(1xn),array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    for k in range(len(thresholdList)):
        threshold = thresholdList[k]
        histogramData = []
        for i in range(binNumber):
            predictionsInBin = []
            for j in range(len(classificationData)):
                if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                        i + 1) * binWidth and testLabels[j] == 0:
                    predictionsInBin.append(predictions[j])
            histogramData.append(positives_count(create_boolean(threshold, predictionsInBin)))

        print(histogramData)

        xData = [minBinValue]
        for i in range(binNumber - 1):
            xData.append(minBinValue + (i + 1) * binWidth)
            xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + binNumber * binWidth)

        yData = []
        for i in range(binNumber):
            yData.append(histogramData[i])
            yData.append(histogramData[i])

        graphTypes = ['-k', "--r", ":b"]
        plt.plot(xData, yData, graphTypes[k % 3])

    print("Preparing Graph")

    plt.title('hadhad: Total positives on background: ' + classificationType)
    plt.xlabel(classificationType)
    plt.xlim(binRange)
    plt.ylabel('positives')
    plt.show()


# Creates a histogram for the total data comparing higgs and QCD

def histogram_1D(binNumber, binRange, testLabels, classificationData, classificationType):
    '''threshold_curve_positives_QCD(int,[int,int],List,array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    histogramDataHiggs = []
    histogramDataTotal = []
    for i in range(binNumber):
        print(i)
        dataInBin = []
        for j in range(len(classificationData)):
            if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                    i + 1) * binWidth:
                dataInBin.append(testLabels[j])
        histogramDataHiggs.append(positives_count(dataInBin))
        histogramDataTotal.append(len(dataInBin))
    print(histogramDataHiggs)
    print(histogramDataTotal)

    xData = [minBinValue]
    for i in range(binNumber - 1):
        xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + (i + 1) * binWidth)
    xData.append(minBinValue + binNumber * binWidth)

    yDataTotal = []
    for i in range(binNumber):
        yDataTotal.append(histogramDataTotal[i])
        yDataTotal.append(histogramDataTotal[i])

    yDataHiggs = []
    for i in range(binNumber):
        yDataHiggs.append(histogramDataHiggs[i])
        yDataHiggs.append(histogramDataHiggs[i])

    plt.plot(xData, yDataTotal, '-k', label="All data")
    plt.plot(xData, yDataHiggs, '-b', label="Higgs")

    print("Preparing Graph")

    plt.title('hadhad: Histogram of total data: ' + classificationType)
    plt.xlabel(classificationType)
    plt.xlim(binRange)
    plt.ylabel('Number of Jets')
    plt.legend(loc="best")
    plt.show()


# Calculates necessary thresholds to DDT for a given threshold and bins

def DDT_thresholds(binNumber, binRange, threshold, predictions, testLabels, classificationData):
    '''DDT_thresholds(int,[int,int],threshold,array(1xn),array(1xn),array(1xn))'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    thresholdList = []
    for i in range(binNumber):
        predictionsInBin = []
        labelsInBin = []
        for j in range(len(classificationData)):
            if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                    i + 1) * binWidth:
                predictionsInBin.append(predictions[j])
                labelsInBin.append(testLabels[j])
        thresholdList.append(percentile_FPR_given_desired([threshold], predictionsInBin, labelsInBin)[0])

    return thresholdList


# Calculates necessary thresholds to DDT with a quadratic fit for a given threshold and bins

def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def DDT_quadratic_thresholds(binNumber, binRange, threshold, predictions, testLabels, classificationData):
    '''DDT_quadratic_thresholds(int,[int,int],threshold,array(1xn),array(1xn),array(1xn))'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    thresholdList = []
    for i in range(binNumber):
        predictionsInBin = []
        labelsInBin = []
        for j in range(len(classificationData)):
            if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                    i + 1) * binWidth:
                predictionsInBin.append(predictions[j])
                labelsInBin.append(testLabels[j])
        thresholdList.append(percentile_FPR_given_desired([threshold], predictionsInBin, labelsInBin)[0])
    xValues = []
    for i in range(binNumber):
        xValues.append(minBinValue + i * binWidth)

    fit, covarianceMatrix = curve_fit(quadratic, xValues, thresholdList, p0=[-1, 1, 0.5])
    thresholdListFitted = []
    for x in xValues:
        thresholdListFitted.append(quadratic(x, fit[0], fit[1], fit[2]))

    return thresholdListFitted


# Creates a graph of binned absolute positives for DDT thesholds

def threshold_curves_DDT(binNumber, binRange, thresholdList, predictions, classificationData, classificationType):
    '''threshold_curves_DDT(int,[int,int],List,array(1xn),array(1xn),str)'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)

    for k in range(len(thresholdList)):
        histogramData = []
        for i in range(binNumber):
            threshold = thresholdList[k][i]
            predictionsInBin = []
            for j in range(len(classificationData)):
                if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                        i + 1) * binWidth:
                    predictionsInBin.append(predictions[j])
            histogramData.append(positives_count(create_boolean(threshold, predictionsInBin)))

        print(histogramData)

        xData = [minBinValue]
        for i in range(binNumber - 1):
            xData.append(minBinValue + (i + 1) * binWidth)
            xData.append(minBinValue + (i + 1) * binWidth)
        xData.append(minBinValue + binNumber * binWidth)

        yData = []
        for i in range(binNumber):
            yData.append(histogramData[i])
            yData.append(histogramData[i])

        graphTypes = ['-k', "--r", ":b"]
        plt.plot(xData, yData, graphTypes[k % 3])

    print("Preparing Graph")

    plt.title('Positives for DDT thresholds: ' + classificationType)
    # plt.title('Histogram of real data: '+classificationType)
    plt.xlabel(classificationType)
    plt.xlim(binRange)
    plt.ylabel('positives')
    plt.show()


# Creates a list of particle count per jet for a given particle type

def particle_count(particleTestData, particleIDs):
    '''particle_count(array(1xn),list)'''
    particleCountsList = []
    for jet in particleTestData:
        particleCount = 0
        for i in range(30):
            if jet[i][9] in particleIDs:
                particleCount += 1
        particleCountsList.append(particleCount)
    return particleCountsList


# Creates a graph of binned absolute positives for many thresholds

def threshold_positives(binNumber, binRange, thresholdList, predictions, classificationData, classificationType,
                        fileName, part, letter):
    '''threshold_positives(int,[int,int],List,array(1xn),array(1xn),str,str,int,"str")'''
    minBinValue = binRange[0]
    maxBinValue = binRange[1]

    binWidth = (maxBinValue - minBinValue) / float(binNumber)
    data = []
    for k in range(len(thresholdList)):
        threshold = thresholdList[k]
        histogramData = []
        for i in range(binNumber):
            predictionsInBin = []
            for j in range(len(classificationData)):
                if classificationData[j] >= minBinValue + i * binWidth and classificationData[j] <= minBinValue + (
                        i + 1) * binWidth:
                    predictionsInBin.append(predictions[j])
            histogramData.append(positives_count(create_boolean(threshold, predictionsInBin)))
        data.append(histogramData)
    data = np.array(data)
    print(data)

    fTwo = h5py.File(fileName, 'a')
    fTwo.create_dataset(str(part) + letter, data=data, compression="lzf")


# Runs functions

jetPT = totalData[trainingDataLength+validationDataLength:, 0:1]
jetMass = totalData[trainingDataLength+validationDataLength:, 1:2]

pTBinRange = [200, 700]
massBinRange = [40, 200]
electronBinRange = [0, 10]
muonBinRange = [0, 10]

desiredFPRs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

'''
thresholdList = percentile_FPR_given_desired(desiredFPRs, predictions, testLabels)
print(thresholdList)
'''

save_ROC_curve(1000, predictions, testLabels, "./data/"+"ROC_curve,"+modelName+".h5")

'''
desiredFPRs=[0.001,0.0001,0.00001]
'''
'''
thresholdList=percentile_FPR_given_desired(desiredFPRs,predictions,testLabels)
'''
'''
threshold_curve_FPR(25,massBinRange,thresholdList,predictions,testLabels,jetMass,"Jet Mass")
'''
'''
threshold_curve_positives_QCD(25,massBinRange,thresholdList,predictions,testLabels,jetMass,"Jet Mass")
'''
'''
histogram_1D(25, massBinRange, testLabels, jetMass, "Jet Mass")

histogram_1D(25, pTBinRange, testLabels, jetpT, "Jet pT")
'''
