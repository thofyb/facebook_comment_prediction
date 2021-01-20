import datetime
import random
import numpy as np
import pandas as pd
import scipy.sparse as sparse

AP_MAX_ITERATIONS = 5
AP_S_KK = -2
TEST_CHECKIN_COUNT = 100
NODE_COUNT = 15000
EDGES_DATA_FILE = "data/Gowalla_edges.txt"
CHECKINS_DATA_FILE = "data/Gowalla_totalCheckins.txt"


def main():
    print("Start time:         %s" % str(datetime.datetime.now()))

    edges = pd.read_csv(EDGES_DATA_FILE, delim_whitespace=True,
                        header=None, names=['A', 'B'])

    graph = sparse.csr_matrix(
        (np.ones(edges.shape[0], dtype=bool),
         (edges['A'], edges['B'])), dtype=bool
    )
    print("Loading done:       %s" % str(datetime.datetime.now()))

    user2Cluster = affinityPropagation(graph.toarray())
    clusters = np.unique(user2Cluster)
    clusterCount = clusters.shape[0]

    print("Cluster count:      %d" % clusterCount)

    info = pd.read_csv(CHECKINS_DATA_FILE, delim_whitespace=True, header=None,
                       names=['UserID', 'Time', 'Latitude', 'Longitude', 'LocationID'])

    users = np.asarray(info['UserID'])
    locations = np.asarray(info['LocationID'])

    recommends = findRecommendations(user2Cluster, clusters, users, locations)

    checkins = [random.randint(0, len(users))
                for _ in range(TEST_CHECKIN_COUNT)]
    checkins = [(users[checkinId],
                 user2Cluster[users[checkinId]],
                 locations[checkinId])
                for checkinId in checkins]

    accuracy = testCheckins(checkins, recommends)
    print("Accuracy [0..100]:  %f" % accuracy)

    print("Done:               %s" % str(datetime.datetime.now()))


def affinityPropagation(S):
    N = NODE_COUNT

    S.flat[::(N + 1)] = AP_S_KK

    R = np.zeros((N, N))
    A = np.zeros((N, N))

    Ids = np.arange(N)

    for _ in range(AP_MAX_ITERATIONS):
        print("  Iteration start:  %s" % str(datetime.datetime.now()))
        AS = A + S

        M0 = AS.max(axis=1)
        amax = AS.argmax(axis=1)

        AS[Ids, amax] = -np.inf
        M1 = AS.max(axis=1)

        R = S - M0

        R[Ids, amax] = S[Ids, amax] - M1
        M2 = np.maximum(R, 0)
        S2 = M2.sum(axis=0)

        for k in range(N):
            A[:, k] = R[k, k] + S2[k] - M2[:, k] - M2[k, k]

        A = np.minimum(A, 0)
        A.flat[::(N + 1)] = M2.sum(axis=0) - M2.diagonal()

    return (A + R).argmax(axis=1)


def findRecommendations(user2Cluster, clusters, users, locations):
    rcmd = {i: {} for i in clusters}

    for user, loc in zip(users, locations):
        c = user2Cluster[user]

        cr = rcmd[c]
        if loc in cr:
            cr[loc] += 1
        else:
            cr[loc] = 1

    result = {}

    for clusterId, clusterDict in rcmd.items():
        srt = [k for k, v in
               sorted(
                   clusterDict.items(),
                   key=lambda i: i[1],
                   reverse=True)]

        result[clusterId] = srt[:10]

    return result


def testCheckins(checkins, recommends):
    count = 0

    for user, cluster, loc in checkins:
        locTopList = recommends[cluster]
        if locTopList is not None:
            for i in locTopList:
                if i == loc:
                    count += 1
                    break
    return count / TEST_CHECKIN_COUNT


main()
