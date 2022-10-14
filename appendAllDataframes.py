import pandas as pd
import numpy as np

<<<<<<< HEAD
def appendTheDataframe():
=======
def appendDf():
>>>>>>> 74abcf709be5f3add72c2664974d0f02260dbf8c
    indexPrincingRequest = []
    pricing_requests = []
    requestHist = np.load("./data/requestHistory.npy")
    for i in range(len(requestHist)):
        if "pricing" in requestHist[i]:
            indexPrincingRequest.append(i)

    print("Number of pricing requests found : ", len(indexPrincingRequest))
    requests = []
<<<<<<< HEAD
=======
    responseHistory = []
>>>>>>> 74abcf709be5f3add72c2664974d0f02260dbf8c
    print("Concatenating all responses to a single dataframe...")
    for index in indexPrincingRequest:
        requests.append(np.load('./data/responseHistory.npy',  allow_pickle=True)[index])

    for r in requests:
        pricing_requests.append(
            pd.DataFrame(r.json()['prices']).assign(**r.json()['request'])
        )
    pricing_requests = pd.concat(pricing_requests)

    print("Concatenating done !")
<<<<<<< HEAD
    print(pricing_requests.head)
=======
>>>>>>> 74abcf709be5f3add72c2664974d0f02260dbf8c
    print("Exporting to csv...")
    pricing_requests.to_csv('./data/allData.csv')
    print("Exporting done !")

if __name__=="__main__":
<<<<<<< HEAD
    appendTheDataframe()
=======
    appendDf()
>>>>>>> 74abcf709be5f3add72c2664974d0f02260dbf8c
