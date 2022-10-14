import numpy as np
import pandas as pd
import appendAllDataframes
import re

def initData():
    np.save('./data/language.npy', ['romanian', 'swedish', 'maltese', 'belgian', 'luxembourgish', 
                    'dutch', 'french', 'finnish', 'austrian', 'slovakian', 'hungarian', 
                    'bulgarian', 'danish', 'greek', 'croatian', 'polish', 'german', 
                    'spanish', 'estonian', 'lithuanian', 'cypriot', 'latvian', 'irish', 
                    'italian', 'slovene', 'czech', 'portuguese'])

    np.save('./data/city.npy', ['vilnius', 'paris', 'valletta', 'madrid', 'amsterdam', 
                'copenhagen', 'rome', 'sofia', 'vienna'])

    np.save('./data/mobile.npy', [0, 1])

    np.save('./data/date.npy', np.linspace(44,0,45))

    np.save('./data/avatar.npy', [])

    np.save('./data/requestHistory.npy', [])

    np.save('./data/responseHistory.npy', [])

def addAvatar(name: str):
    # Add an avatar if not already existing
    np.save('./data/avatar.npy', list(dict.fromkeys(np.append(np.load('./data/avatar.npy'),name))))

def addRequest(req: str):
    np.save('./data/requestHistory.npy', (np.append(np.load('./data/requestHistory.npy'),req)))

def addResponseHistory(resp):
    np.save('./data/responseHistory.npy', (np.append(np.load('./data/responseHistory.npy', allow_pickle=True),resp)))

def getAllAvatar():
    return np.load("./data/avatar.npy")

def getAllRequests():
    return np.load("./data/requestHistory.npy")

def getAllResponses():
    return np.load("./data/responseHistory.npy", allow_pickle=True)

def getAllCity():
    return np.load("./data/city.npy")

def getAllLanguage():
    return np.load("./data/language.npy")

def getAllDate():
    return np.load("./data/date.npy")

def deleteLastRequest():
    np.save('./data/requestHistory.npy', getAllRequests()[0:len(getAllRequests())-1])
    np.save('./data/responseHistory.npy', getAllResponses()[0:len(getAllResponses())-1])
    print("Deleted last request !")

def createAvatarIDcsv():
    req = getAllRequests()
    resp = getAllResponses()
    listID = []
    listName = []
    for i in range(len(req)):
        if "nameAvatar" in req[i]:
            # Alors c'est le bordel mais en gros ca recupere le texte apres "id", 
            # puis on garde juste les chiffres avant un autre caractere
            # grace a cette belle expression reguliere  et ca donne l'ID c'est fabuleux
            # Puis on met un indice 0 a la fin psk cest une liste et qu'on veut que le 1er elem
            # et on le cast en integer pour avoir le bon format et GG
            listID.append(int(re.findall(r"(\d+)[,}]", resp[i].text.split('"id":',1)[1])[0]))
            # Alors la en gros on fait pareil mais on split deux fois et hopla magie
            # On recupere le nom
            listName.append(resp[i].text.split('"name":"',1)[1].split('"',1)[0])
    # Et on finit par exporter le bordel en csv
    pd.DataFrame({"avatar_name":listName, "avatar_id":listID}).to_csv("./data/AvatarNameAndID.csv")
    print("AvatarNameAndID.csv saved !")

def getMinDayOfAvatar(avatarName):
    appendAllDataframes.appendDf()
    dfAvatarID = pd.read_csv('./data/AvatarNameAndID.csv')
    df = pd.read_csv("./data/allData.csv")
    df["avatar_id"] = pd.to_numeric(df["avatar_id"])
    # On recupere l'ID associe, au nom de l'avatar puis on le met en string et on garde a partir du 5eme carac
    # parce que c'est mal fichu ce bordel et on cast en int
    idAvatar = int(dfAvatarID.loc[dfAvatarID["avatar_name"]== avatarName, "avatar_id"].to_string()[5:])

    # CA MARCHE PAS COMPREND PAS PK NIQUE SA MERE
    # df = df[df["avatar_id"]==idAvatar]
    # print(df)
    #TODO: link avatar to its ID
getMinDayOfAvatar("Avataricard02")

def rearrangeCol(bonOrdre, aFaire):
    listColOrdre = bonOrdre.columns.tolist()
    df = pd.DataFrame()
    for col in listColOrdre:
        df[col] = aFaire[col].tolist()
    return df


# Add the request_order column to the inputted dataframe
def addOrderRequest(df):
    df = df.rename(columns={"Unnamed: 0":"ind"})
    avatarList = pd.unique(df['avatar_id'])
    cptOrdReq = {}
    for avatar in avatarList:
        cptOrdReq[avatar] = 1
    addOne = False
    for ind, row in df.iterrows():
        if df.loc[ind, "ind"] > 0:
            addOne = True
        if addOne and df.loc[ind, "ind"] == 0:
            cptOrdReq[row["avatar_id"]] = cptOrdReq[row["avatar_id"]] +1
            addOne = False
        df.loc[ind, "order_requests"] = cptOrdReq[row["avatar_id"]]
    # Delete the previous index column that is now useless
    df = df.drop(['ind'], axis=1)
    return df



# Add the hotel features on the inputted dataframe
def prepareDataframe(df):
    hotels = pd.read_csv('./data/features_hotels.csv', index_col=['hotel_id', 'city'])
    return df.join(hotels, on=['hotel_id', 'city'])



def main(doInit = False):
    if doInit: 
        #Validation part
        while True:
            try:
                inputVal = input("Are you sure you want to initialize all the data (it will erase all the previous) (y/n): ")
            except ValueError:
                print("Sorry, I didn't understand that.")
                #better try again... Return to the start of the loop
                continue

            if inputVal == "y":
                break
            if inputVal == "n":
                doInit = False
                break
            else:
                print("Sorry, I didn't understand that.")
                continue
        # If validated then do it
        if doInit:
            initData()

if __name__=="__main__":
    main(False)