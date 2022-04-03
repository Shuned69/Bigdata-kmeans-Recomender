#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd
import numpy as np
from math import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse

def findMax(list):
    max = 0
    for i in list:
        if max < ceil(i[0]):
            max = ceil(i[0])
    return max

def createSparseMatrice(test, max):
    newTab = np.zeros([len(test), max])
    for i in  range(len(test)):
        newTab[i,ceil(test[i][0] - 1)] = 1
    return newTab


def getItemPrice(cli_id, customersData):
    if (cli_id in customersData):
        tmp = customersData[cli_id]["order_price_average"]
        return (tmp)
    return (False)


def getnumberItems(cli_id, customersData):
    if (cli_id in customersData):
        tmp = customersData[cli_id]["nb_products"]
        return (tmp)
    return (False)

def parseData(cli_id, user):
    tmpuser = user[['CLI_ID']]
    for i in range(len(tmpuser)):
        if (user.iloc[i]['CLI_ID']) == int(cli_id):
            return (i)
    return "error"

def recommandItem(targetId):
    print("Creating Price Based Clusters")

    data = pd.read_csv("./data/maille_Kado.csv")

    user = data.sort_values(by=['CLI_ID'], ascending=[True])
    user = user.drop_duplicates(subset=['CLI_ID'])
    newLen = len(user)
    userIdList = [i+1 for i in range(newLen)]
    users = np.unique(userIdList)
    user['USER_ID'] = userIdList
    user = user[['USER_ID', 'CLI_ID']]

    import json
    from urllib.request import urlopen, Request

    requestCustomers = Request('https://romainpiot.fr/bigdata/customers.json', headers={'User-Agent': 'Mozilla/5.0'})
    customersData = json.loads(urlopen(requestCustomers).read())

    averagePrice = []
    numberItems = []

    for userId in user['CLI_ID']:
        price = getItemPrice(str(userId), customersData)
        if price != False:
            averagePriceTmp = []
            averagePriceTmp.append(float(price))
            averagePriceTmp.append(userId)
            averagePrice.append(averagePriceTmp)
        else:
            break
        items = getnumberItems(str(userId), customersData)
        if items != False:
            numberItemsTmp = []
            numberItemsTmp.append(float(items))
            numberItemsTmp.append(userId)
            numberItems.append(numberItemsTmp)

    X = np.array(averagePrice)
    Y = np.array(numberItems)


    maxPrice = findMax(X)
    maxItems = findMax(Y)

    sparcePrice = createSparseMatrice(X, maxPrice)
    sparceItems = createSparseMatrice(Y, maxItems)

    df_sparseMatrixPrice = pd.DataFrame(sparcePrice, index = [i+1 for i in range(len(user))], columns= [i+1 for i in range(maxPrice)] )

    df_sparseMatrixItems = pd.DataFrame(sparceItems, index = [i+1 for i in range(len(user))], columns= [i+1 for i in range(maxItems)] )

    sseX = calculate_WSS(X, 10)
    sseY = calculate_WSS(Y, 10)

    plt.plot(sseX)
    plt.show()

    plt.plot(sseY)
    plt.show()

    silPrice = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
      kmeans = KMeans(n_clusters = k).fit(df_sparseMatrixPrice)
      labels = kmeans.labels_
      silPrice.append(silhouette_score(df_sparseMatrixPrice, labels, metric = 'euclidean'))
    plt.plot(silPrice)
    plt.show()

    silItems = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
      kmeans = KMeans(n_clusters = k).fit(df_sparseMatrixItems)
      labels = kmeans.labels_
      silItems.append(silhouette_score(df_sparseMatrixItems, labels, metric = 'euclidean'))
    plt.plot(silItems)
    plt.show()

    kmeansPrice = KMeans(n_clusters=7, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    clustersPrice = kmeansPrice.fit_predict(df_sparseMatrixPrice)
    kmeansItems = KMeans(n_clusters=5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    clustersItems = kmeansItems.fit_predict(df_sparseMatrixItems)

    users = np.array([i+1 for i in range(len(user))])
    clid = np.array(user['CLI_ID'][:len(user)])

    users_clusterPrice = pd.DataFrame(np.concatenate((users.reshape(-1,1),clid.reshape(-1,1), clustersPrice.reshape(-1,1)), axis = 1), columns = ['userId', 'cli_id', 'Cluster'])

    plt.scatter(X[:,0],X[:,1], c=kmeansPrice.labels_, cmap='rainbow')

    users = np.array([i+1 for i in range(len(user))])
    clid = np.array(user['CLI_ID'][:len(user)])

    users_clusterItems = pd.DataFrame(np.concatenate((users.reshape(-1,1),clid.reshape(-1,1), clustersItems.reshape(-1,1)), axis = 1), columns = ['userId', 'cli_id', 'Cluster'])
    plt.scatter(Y[:,0],Y[:,1], c=kmeansItems.labels_, cmap='rainbow')

    cid = parseData(targetId, user)
    if cid == "error":
           print(f"Could not find client" )
    else:
           tmp = clustersPrice[cid]


    sameClusterUser = []
    for i in range(len(clustersPrice)):
           if clustersPrice[i] == tmp:
                  sameClusterUser.append(i)


    newData = []

    scuList = []
    for scu in sameClusterUser:
        scuList.append(user["CLI_ID"].iloc[scu])


    dfNewData = data.loc[data["CLI_ID"].isin(scuList)]

    dfNewData.to_csv("./data/price_Kado.csv")
    # for i in range(len(user)):
    #        for j in sameClusterUser:
    #               if data.iloc[i]['CLI_ID'] == user.iloc[j]['CLI_ID']:
    #                   if data.iloc[i]['CLI_ID'] != targetId:
    #                      newData.append(data.iloc[i])
    #
    # dfNewData = pd.DataFrame(newData)
    #
    # dfNewData.to_csv("./data/price_Kado.csv")

    kdoData = pd.read_csv("./data/price_Kado.csv")

    rand = random.randint(0, len(kdoData)-1)

    # kdoData = kdoData[['LIBELLE']]
    gift = kdoData.iloc[rand]['LIBELLE']
    based = kdoData.iloc[rand]['CLI_ID']
    print(f"Client", targetId, "will be gifted", gift, "based on him belonging to the sames purchases behavior (meshes, price average, number of items bought) clusters than ",based)
    return (gift)
