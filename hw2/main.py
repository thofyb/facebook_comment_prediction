import random
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from scipy import sparse
from scipy.sparse import hstack, coo_matrix, csr_matrix
from math import sqrt

FACTOR_COUNT = 2


def crossValidate():
    data_file_path = "./dataset.csv"
    df = pd.read_csv(data_file_path, header=None, names=['UserId', 'Rating', 'MovieId'])

    instance_count, _ = df.shape
    ones_column = coo_matrix(np.full((instance_count, 1), 1))

    encoder = OneHotEncoder(categories='auto')
    one_hot_user_matrix = encoder.fit_transform(np.asarray(df['UserId']).reshape(-1, 1))
    one_hot_movie_matrix = encoder.fit_transform(np.asarray(df['MovieId']).reshape(-1, 1))

    X = hstack([ones_column, one_hot_user_matrix, one_hot_movie_matrix]).tocsr()
    ratings = csr_matrix(np.asarray(df['Rating']).reshape(-1, 1))

    X, ratings = shuffle(X, ratings)

    instanceCount, featureCount = X.shape

    foldCount = 5
    folds = []
    step = instanceCount // foldCount
    for i in range(foldCount - 1):
        folds.append((i * step, (i + 1) * step - 1))
    folds.append(((foldCount - 1) * step, instanceCount - 1))

    resultTable = pd.DataFrame(
        columns=[],
        index=["R^2", "RMSE", "R^2-train", "RMSE-train"]
    )

    for i in range(foldCount):
        r2, rmse, r2_tr, rmse_tr = gradientDescent(X, folds[i], ratings)
        resultTable.insert(i, "T%i" % (i + 1), np.array([r2, rmse, r2_tr, rmse_tr]))

    e = resultTable.mean(axis=1)
    std = resultTable.std(axis=1)
    resultTable.insert(5, "E", e)
    resultTable.insert(6, "STD", std)

    print(resultTable.head())

    resultTable.to_csv("out.csv")


def gradientDescent(allInstances, testRangeTuple, results):
    instanceCount, featureCount = allInstances.shape
    weights_k = weights_prev = sparse.rand(featureCount, 1, 0.001).tocsr()
    V_k = V_prev = sparse.rand(featureCount, FACTOR_COUNT, 0.001).tocsr()
    max_iterations = 6

    for i in range(1, max_iterations):
        lambda_i = 0.1 / i

        randIds = [getPosition(instanceCount, testRangeTuple) for _ in range(1000)]

        w_grad, V_grad = getMSEGradient(randIds, allInstances, weights_k, V_k, results)

        weights_k = weights_prev - lambda_i * w_grad
        V_k = V_prev - lambda_i * V_grad

        weights_prev = weights_k
        V_prev = V_k

    r2_train, r2_test = getR2(allInstances, weights_k, V_k, results, testRangeTuple)
    rmse_train, rmse_test = getRMSE(allInstances, weights_k, V_k, results, testRangeTuple)

    return r2_test, rmse_test, r2_train, rmse_train


def getMSEGradient(instanceIDs, instances, weights, V, results):
    instanceCount, featureCount = instances.shape

    A0, B1, W1, predicted = getMatricies(instances, V, weights)

    w_R = sparse.csr_matrix((1, featureCount))
    for a in instanceIDs:
        w_R += (results[a, 0] - predicted[a, 0]) * (instances[a, :])
    w_R = 2.0 / len(instanceIDs) * w_R
    w_R = w_R.transpose()

    II = instances.multiply(instances).transpose()

    V_R = sparse.lil_matrix((featureCount, FACTOR_COUNT))
    for v_f in range(FACTOR_COUNT):
        for a in instanceIDs:
            s = instances[a, :].transpose() * A0[a, v_f] - V[:, v_f].multiply(II[:, a])
            V_R[:, v_f] = (results[a, 0] - predicted[a, 0]) * s

    V_R = 2.0 / len(instanceIDs) * V_R

    return w_R.tocsr(), V_R.tocsr()


def getR2(instances, weights, V, results, testRangeTuple):
    instanceCount, featureCount = instances.shape
    ta = testRangeTuple[0]
    tb = testRangeTuple[1]

    A0, B1, W1, predicted = getMatricies(instances, V, weights)

    yTest_avg = results[ta:tb, 0].mean()

    D0Test = results[ta:tb, 0] - predicted[ta:tb, 0]
    aTest = D0Test.multiply(D0Test).sum()
    D1Test = results[ta:tb, 0] - sparse.csr_matrix([yTest_avg for _ in range(tb - ta)]).transpose()
    bTest = D1Test.multiply(D1Test).sum()

    if ta == 0:
        yTrain_avg = results[(tb + 1):, 0].mean()

        D0Train = results[(tb + 1):, 0] - predicted[(tb + 1):, 0]
        aTrain = D0Train.multiply(D0Train).sum()

        D1Train = results[(tb + 1):, 0] - sparse.csr_matrix(
            [yTrain_avg for _ in range(instanceCount - tb - 1)]).transpose()
        bTrain = D1Train.multiply(D1Train).sum()

    elif tb == instanceCount - 1:
        yTrain_avg = results[:(ta - 1), 0].mean()

        D0Train = results[:(ta - 1), 0] - predicted[:(ta - 1), 0]
        aTrain = D0Train.multiply(D0Train).sum()

        D1Train = results[:(ta - 1), 0] - sparse.csr_matrix([yTrain_avg for _ in range(ta - 1)]).transpose()
        bTrain = D1Train.multiply(D1Train).sum()

    else:
        yTrain_avg = (results[(tb + 1):, 0].mean() + results[:(ta - 1), 0].mean()) * 0.5
        aTrain = getSumTrain(results, predicted, ta, tb)

        D1_0Train = results[:(ta - 1), 0] - sparse.csr_matrix([yTrain_avg for _ in range(ta - 1)]).transpose()
        D1_1Train = results[(tb + 1):, 0] - sparse.csr_matrix(
            [yTrain_avg for _ in range(instanceCount - tb - 1)]).transpose()
        bTrain = D1_0Train.multiply(D1_0Train).sum() + D1_1Train.multiply(D1_1Train).sum()

    return 1 - aTrain / bTrain, 1 - aTest / bTest


def getRMSE(instances, weights, V, results, testRangeTuple):
    instanceCount, featureCount = instances.shape
    ta = testRangeTuple[0]
    tb = testRangeTuple[1]

    A0, B1, W1, predicted = getMatricies(instances, V, weights)

    D0Test = results[ta:tb, 0] - predicted[ta:tb, 0]
    sumTest = D0Test.multiply(D0Test).sum()

    if ta == 0:
        D0Train = results[(tb + 1):, 0] - predicted[(tb + 1):, 0]
        sumTrain = D0Train.multiply(D0Train).sum()

    elif tb == instanceCount - 1:
        D0Train = results[:(ta - 1), 0] - predicted[:(ta - 1), 0]
        sumTrain = D0Train.multiply(D0Train).sum()

    else:
        sumTrain = getSumTrain(results, predicted, ta, tb)

    return sqrt(1 / instanceCount * sumTrain), sqrt(1 / instanceCount * sumTest)


def getSumTrain(results, predicted, ta, tb):
    D0_0Train = results[:(ta - 1), 0] - predicted[:(ta - 1), 0]
    D0_1Train = results[(tb + 1):, 0] - predicted[(tb + 1):, 0]
    return D0_0Train.multiply(D0_0Train).sum() + D0_1Train.multiply(D0_1Train).sum()


def getMatricies(instances, V, weights):
    A0 = instances * V
    A1 = A0.multiply(A0)
    sqV = V.multiply(V)
    sqX = instances
    A2 = sqX * sqV
    B0 = A1 - A2
    B1 = B0 * sparse.csr_matrix([1 for _ in range(FACTOR_COUNT)]).transpose()
    W1 = instances * weights
    predicted = W1 + B1 * 0.5
    return A0, B1, W1, predicted


def getPosition(count, notInRange):
    a, b = notInRange
    r = random.randint(0, count - (b + 1 - a) - 1)
    if r >= a:
        r += b + 1 - a
    return r


crossValidate()
