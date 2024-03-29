import pandas as pd
import numpy as np
import features

def appendDf():
    indexPrincingRequest = []
    pricing_requests = []
    requestHist = np.load("./data/requestHistory.npy")
    for i in range(len(requestHist)):
        if "pricing" in requestHist[i]:
            indexPrincingRequest.append(i)

    print("Number of pricing requests found : ", len(indexPrincingRequest))
    requests = []
    print("Concatenating all responses to a single dataframe...")
    resp = np.load('./data/responseHistory.npy',  allow_pickle=True)
    for index in indexPrincingRequest:
        requests.append(resp[index])

    for r in requests:
        pricing_requests.append(
            pd.DataFrame(r.json()['prices']).assign(**r.json()['request'])
        )
    pricing_requests = pd.concat(pricing_requests)

    print("Concatenating done !")
    pricing_requests.to_csv('./data/allData.csv')

    print("Adding order requests")
    df = pd.read_csv('./data/allData.csv')
    print(df.dtypes)
    # df.drop(["Unnamed: 0"], axis=1, inplace=True)
    pricing_requests = features.addOrderRequest(df)
    print("Exporting to csv...")
    pricing_requests.to_csv('./data/allData.csv')

    print("Exporting done !")

if __name__=="__main__":
    appendDf()
