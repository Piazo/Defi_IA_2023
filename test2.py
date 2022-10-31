import features
import re
import pandas as pd

def addOrderRequest(df):
    df = df.rename(columns={"Unnamed: 0":"ind"})
    avatarList = pd.unique(df['avatar_id'])
    cptOrdReq = {}
    for avatar in avatarList:
        cptOrdReq[avatar] = 1
    lastAvatarID = 0
    addOne = False
    for ind, row in df.iterrows():
        if df.loc[ind, "ind"] > 0:
            addOne = True
        if addOne and df.loc[ind, "ind"] == 0 and lastAvatarID == df.loc[ind, "avatar_id"]:
            cptOrdReq[row["avatar_id"]] = cptOrdReq[row["avatar_id"]] +1
            addOne = False
        df.loc[ind, "order_requests"] = cptOrdReq[row["avatar_id"]]
        lastAvatarID = df.loc[ind, "avatar_id"]
    # Delete the previous index column that is now useless
    df = df.drop(['ind'], axis=1)
    return df

def main():
    df = pd.read_csv("./data/allData.csv")
    df = addOrderRequest(df)
    df.to_csv("requestOrderOK.csv")

main()