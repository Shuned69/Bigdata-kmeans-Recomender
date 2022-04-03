import clusterMaille as cm
import customerPrice as cp
import pandas as pd
import json
from urllib.request import urlopen, Request
import sys

data = pd.read_csv("./data/KaDo_clean.csv")

requestCustomers = Request('https://romainpiot.fr/bigdata/customers.json', headers={'User-Agent': 'Mozilla/5.0'})
customersData = json.loads(urlopen(requestCustomers).read())

def get_row_per_customer_list(df, list=[]):
    list = [int(x) for x in list]
    return df.loc[df["CLI_ID"].isin(list)]

def getSegmType(targetId):
    tab = []
    for key in customersData:
        if (targetId == key):
            tmpType = customersData[targetId]["segm_buyer_type"]
            tmpRec = customersData[targetId]["segm_recurrent_buy_product"]
            tab.append(tmpType)
            tab.append(tmpRec)
            return (tab)
    return (False)


def segmentDataSet(targetId):
    print("segmenting data")

    sgm = getSegmType(targetId)
    if sgm == False:
        print("error: client id not found")
        return "error: client id not found"
    users = []
    for key in customersData:
        if (customersData[key]["segm_buyer_type"] == str(sgm[0])) and (customersData[key]["segm_recurrent_buy_product"] == str(sgm[1])):
            users.append(key)

    newdf = get_row_per_customer_list(data,users)
    newdf.to_csv("./data/segm_Kado.csv")
    print("done")
    return 1


def main(targetId):
    if (segmentDataSet(targetId) == 1):
        cm.createMailleCluster(targetId)
        gift = cp.recommandItem(targetId)
    else:
         exit(84)
    print (customersData[targetId])

if __name__ == "__main__":
    main(sys.argv[1])







