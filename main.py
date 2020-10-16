import pandas as pd
import numpy as np
from math import sqrt


def crossValidate():
    featureNames = getFeatureNames()
    linFeatureNames = getLinFeatureNames()

    dataset = pd.read_csv("dataset.csv", names=featureNames + ["Result"])

    sets = np.array_split(dataset, 5)

    for variant in sets:
        for feature in featureNames:
            value = variant[feature].max() - variant[feature].min()
            if value != 0:
                variant[feature] = (variant[feature] - variant[feature].min()) / value
        for lfn in linFeatureNames:
            del variant[lfn]

    resultTable = pd.DataFrame(
        columns=[],
        index=["R^2", "RMSE", "R^2-train", "RMSE-train"] + [f for f in featureNames if f not in linFeatureNames]
    )

    for i in range(len(sets)):
        testSet = sets[i]
        trainSet = pd.concat([
            sets[j] for j in range(len(sets)) if j != i
        ])

        weights, r2, rmse, r2_t, rmse_t = gradientDescent(trainSet, testSet)
        resultTable.insert(i, "T" + str(i + 1), np.concatenate((np.array([r2, rmse, r2_t, rmse_t]), weights[1:])))

    e = resultTable.mean(axis=1)
    std = resultTable.std(axis=1)
    resultTable.insert(5, "E", e)
    resultTable.insert(6, "STD", std)

    print(resultTable.head())

    resultTable.to_csv("results.csv")


def gradientDescent(trainSet, testSet):
    maxIterations = 100

    instances = trainSet.to_numpy()
    featureCount = len(trainSet.columns) - 1
    results = instances[:, featureCount]
    instances = np.delete(instances, featureCount, axis=1)
    instances = np.insert(instances, 0, 1, axis=1)
    featureCount += 1

    w0 = np.full(featureCount, 1)
    wk = wk_prev = w0
    k = 1

    while k < maxIterations:
        lambda_k = 1 / k
        gradient = np.array([
            partialDerivativeMSE(j, instances, wk_prev, results)
            for j in range(0, featureCount)])

        wk = wk_prev - lambda_k * gradient

        wk_prev = wk
        k += 1

    r2_train = getR2(instances, wk, results)
    rmse_train = getRMSE(instances, wk, results)

    instances_t = testSet.to_numpy()
    featureCount_t = len(testSet.columns) - 1
    results_t = instances_t[:, featureCount_t]
    instances_t = np.delete(instances_t, featureCount_t, axis=1)
    instances_t = np.insert(instances_t, 0, 1, axis=1)

    r2_t = getR2(instances_t, wk, results_t)
    rmse_t = getRMSE(instances_t, wk, results_t)

    return wk, r2_t, rmse_t, r2_train, rmse_train


def partialDerivativeMSE(index, instances, weights, results):
    instanceCount, featureCount = instances.shape
    s = 0

    for i in range(instanceCount):
        inst = instances[i]
        result = results[i]
        d = result - np.dot(inst, weights)
        s += d * (-inst[index])

    return 2 / instanceCount * s


def getR2(instances, weights, results):
    instanceCount, featureCount = instances.shape
    a = b = 0
    y_avg = np.average(results)

    for i in range(instanceCount):
        p = np.dot(instances[i], weights)
        y = results[i]
        a += (y - p) ** 2
        b += (y - y_avg) ** 2

    return 1 - a / b


def getRMSE(instances, weights, results):
    instanceCount, featureCount = instances.shape
    s = 0

    for i in range(instanceCount):
        p = np.dot(instances[i], weights)
        y = results[i]
        s += pow(y - p, 2)

    return sqrt(1 / instanceCount * s)


def getFeatureNames():
    page = [
        "Likes",
        "Checkin",
        "Talks",
        "Category",
    ]

    essential = [
        "CC1",
        "CC2",
        "CC3",
        "CC4",
        "CC5",
    ]

    derived = [c + " @" + d for c in essential for d in ["Min", "Max", "Avg", "Med", "Std"]]

    other = [
        "Base time",
        "Post Length",
        "Post Share Count",
        "Post Promotion Status",
        "Target Hours"
    ]

    weekdays = ["Post published @" + w for w in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]]
    weekdays += ["Base date/time @" + w for w in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]]

    return page + derived + essential + other + weekdays


def getLinFeatureNames():
    return [
        "Post published @Sat",
        "Base date/time @Sat",
        "CC1 @Avg",
        "CC2 @Avg",
        "CC3 @Avg",
        "CC4 @Avg",
        "CC5 @Avg"
    ]


crossValidate()
