#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os
import json


def createSparseMatrice(test):
       newTab = np.zeros([len(test), 34])
       for i in range(len(test)):
              for j in range(len(test[i])):
                     newTab[i, test[i][j] - 1] = 1
       return newTab

def createMailleCluster(targetId):
       print("Creating Maille Based Clusters")
       data = pd.read_csv("./data/segm_Kado.csv")
       dataRange = len(data)
       print ('Data Loaded.'
              'Now creating list...')

       maille = data.sort_values(by=['FAMILLE', 'UNIVERS', 'MAILLE'], ascending=[True, True, True])
       maille = maille.drop_duplicates(subset=['MAILLE'])
       mlen = len(maille)
       idList = [i+1 for i in range(mlen)]
       maille['PRODUCT_ID'] = idList
       maille = maille[['MAILLE','PRODUCT_ID']]

       user = data.sort_values(by=['CLI_ID'], ascending=[True])
       user = user.drop_duplicates(subset=['CLI_ID'])
       newLen = len(user)
       userIdList = [i+1 for i in range(newLen)]
       user['USER_ID'] = userIdList
       user = user[['USER_ID', 'CLI_ID']]

       print ('List created.'
              'Now creating Array...')

       test = []
       tmp_user = []
       tmp_merch = []
       tmp_merch2 = []
       dataLen = len(data)


       for i in range(dataRange):
              tmp = (maille[maille['MAILLE'] == (data.iloc[i]['MAILLE'])])
              tmp_m = tmp.iloc[0]['PRODUCT_ID']
              tmp2 = (user[user['CLI_ID'] == (data.iloc[i]['CLI_ID'])])
              tmp_u = tmp2.iloc[0]['USER_ID']
              if tmp_u in tmp_user:
                     if (tmp_m in tmp_merch[-1]) == False:
                            tmp_merch[-1].append(int(tmp_m))
              else:
                     tmp_merch2 = []
                     tmp_user.append(tmp_u)
                     tmp_merch2.append(tmp_m)
                     tmp_merch.append(tmp_merch2)

       for i in range(len(tmp_user)):
              test.append([tmp_user[i], tmp_merch[i]])



       var = createSparseMatrice(tmp_merch)
       np.shape(var)

       df_sparseMatrix = pd.DataFrame(var, index = [i+1 for i in range(len(tmp_merch))], columns=[i+1 for i in range(34)])

       kmeans = KMeans(n_clusters=6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
       clusters = kmeans.fit_predict(df_sparseMatrix)

       users = np.array([i+1 for i in range(len(tmp_merch))])
       clid = np.array(user['CLI_ID'][:len(tmp_merch)])


       users_cluster = pd.DataFrame(np.concatenate((users.reshape(-1,1),clid.reshape(-1,1), clusters.reshape(-1,1)), axis = 1), columns = ['userId', 'cli_id', 'Cluster'])

       def parseData(cli_id):

              tmpuser= user[['CLI_ID']]
              for i in range(len(tmpuser)):
                     if (user.iloc[i]['CLI_ID']) == int(cli_id):
                            return (i)
              return "error"


       cid = parseData(targetId)
       if cid == "error":
              print(f"Could not find client" )
       else:
              tmp = clusters[cid]


       sameClusterUser = []
       for i in range(len(clusters)):
              if clusters[i] == tmp:
                     sameClusterUser.append(i)


       scuList = []
       for scu in sameClusterUser:
              scuList.append(user["CLI_ID"].iloc[scu])


       dfNewData = data.loc[data["CLI_ID"].isin(scuList)]

       dfNewData.to_csv("./data/maille_Kado.csv")

       # for i in range(rng):
       #        if (i % 1000) == 1:
       #               print(i, "/", dataRange)
       #        for j in sameClusterUser:
       #               if data.iloc[i]['CLI_ID'] == user.iloc[j]['CLI_ID']:
       #                      newData.append(data.iloc[i])
       # for i in range(rng):
       #        if (i % 1000) == 1:
       #               print(rng + i, "/", dataRange)
       #        for j in sameClusterUser:
       #               if data.iloc[i + rng]['CLI_ID'] == user.iloc[j]['CLI_ID']:
       #                      newData.append(data.iloc[i])
       #
       # dfNewData = pd.DataFrame(newData)
       # dfNewData.to_csv("./data/maille_Kado.csv")





