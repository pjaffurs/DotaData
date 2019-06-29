import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
import csv

# dotaProject.py
# by Patterson Jaffurs
# COEN 129: Machine Learning and Data Mining
# Dota 2 Personal Project
# Examines a large set of data on results of Dota 2 games and compares 
# different machine learning algorithms to discover which can mest model
# the trend between heroes chosen, game mode, and game type in regards 
# to winner.

def getWinrate(data, results, hero):
    """
    getWinrate()
    Calculates the win-rate of a given hero, based
    on their ID.
    """
    gameCount = 0
    wins = 0.0
    for game in data:
        if game[hero + 2] == 0:
            continue
        elif game[hero + 2] == results[gameCount]:
            wins += 1
        gameCount += 1
    return float(wins / gameCount)
        
############################## CODE BODY ##############################
# read in training data
trainingX = []
trainingY = []
with open('dota2Train.csv', 'r') as data:
    reader = csv.reader(data, delimiter=',')
    for line in reader:
        trainingY.append(float(line[0]))
        line.pop(0)
        line.pop(0) #remove region
        x = np.array([float(v) for v in line])
        trainingX.append(x)

trainingX = np.array(trainingX)
trainingY = np.array(trainingY)

print('Finished reading training data')
# read in test data
testX = []
testY = []
with open('dota2Test.csv', 'r') as data:
    reader = csv.reader(data, delimiter=',')
    for line in reader:
        testY.append(float(line[0]))
        line.pop(0)
        line.pop(0) #remove region
        x = np.array([float(v) for v in line])
        testX.append(x)

testX = np.array(testX)
testY = np.array(testY)

print('Finished reading test data')

# preprocessing stage
# final version uses l2 normalization
pre = sklearn.preprocessing.Normalizer(norm='l2')
#trainingX = sklearn.preprocessing.scale(trainingX)
#testX = sklearn.preprocessing.scale(testX)
normX = pre.fit_transform(trainingX)
testX = pre.transform(testX)
print('Finished preprocessing')

# 1) Attempt Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(normX, trainingY)
print("LDA fitted")

ldaResults = lda.predict(testX)
ldaRMSE = sklearn.metrics.mean_squared_error(testY, ldaResults) ** 0.5
print('LDA RMSE: {}'.format(ldaRMSE))
print('LDA Score: {}'.format(lda.score(testX, testY)))

# 2) Attempt Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(normX, trainingY)
print('QDA fitted')

qdaResults = qda.predict(testX)
print(np.sum(qdaResults == testY))
qdaRMSE = sklearn.metrics.mean_squared_error(testY, qdaResults) ** 0.5
print('QDA RMSE: {}'.format(qdaRMSE))
print('QDA Score: {}'.format(qda.score(testX, testY)))

# 3) Attempt Logistic Regression
logReg = sklearn.linear_model.LogisticRegression()
logReg.fit(normX, trainingY)
print('LogReg fitted')

logRegResults = logReg.predict(testX)
logRegRMSE = sklearn.metrics.mean_squared_error(testY, logRegResults) ** 0.5
print('LogReg RMSE: {}'.format(logRegRMSE))
print('LogReg Score: {}'.format(logReg.score(testX, testY)))

# 4) Attempt Naive Bayes
nb = BernoulliNB()
nb.fit(normX, trainingY)
print('Naive Bayes fitted')

nbResults = nb.predict(testX)
nbRMSE = sklearn.metrics.mean_squared_error(testY, nbResults) ** 0.5
print('Naive Bayes RMSE: {}'.format(nbRMSE))
print('Naive Bayes Score: {}'.format(nb.score(testX,testY)))

# 5) Attempt SGD log reg
sgd = sklearn.linear_model.SGDClassifier(loss='log')
sgd.fit(normX, trainingY)
print('SGD fitted')

sgdResults = sgd.predict(testX)
sgdRMSE = sklearn.metrics.mean_squared_error(testY, sgdResults) ** 0.5
print('SGD RMSE: {}'.format(sgdRMSE))
print('SGD Score: {}'.format(sgd.score(testX, testY)))

# 6) Attempt KNN takes 100 millenia to finish, same RMSE as SGD
# DOES NOT WORK, WILL CRASH, DATASET TOO LARGE
#knn = sklearn.neighbors.KNeighborsClassifier()
#knn.fit(trainingX, trainingY)
#print('KNN fitted')

#knnResults = knn.predict(testX)
#knnRMSE = sklearn.metrics.mean_squared_error(testY, sgdResults) ** 0.5
#print('KNN RMSE: {}'.format(knnRMSE))

# Data Analysis
# Hero Winrates
print('Anti-Mage Winrate: {}'.format(getWinrate(trainingX, trainingY, 1)))

# hero 23 and 107 are all zeroes and must be omitted
totalWinrate = [getWinrate(trainingX, trainingY, i) for i in [x for x in range(113) if x != 23 and x != 107]]
for i in range(111):
    print('{}: {}'.format(i, totalWinrate[i]))
print(np.mean(totalWinrate))
