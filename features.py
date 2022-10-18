import numpy as np
import pandas as pd
import appendAllDataframes
import re
import streamlit as st
import random
import math
import plotly.express as px
import makeRequest


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
    np.save('./data/requestHistory.npy', getAllRequests()[0:len(getAllRequests())-200])
    np.save('./data/responseHistory.npy', getAllResponses()[0:len(getAllResponses())-200])
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


def getAvatarName(id):
    dfAvatarID = pd.read_csv('./data/AvatarNameAndID.csv')
    return pd.unique(dfAvatarID[dfAvatarID["avatar_id"] == id]["avatar_name"])[0]

def getMinDayOfAvatar(avatarName):
    dfAvatarID = pd.read_csv('./data/AvatarNameAndID.csv')
    df = pd.read_csv("./data/allData.csv")
    # On recupere l'ID associe, au nom de l'avatar puis on le met en string et on garde a partir du 5eme carac
    # parce que c'est mal fichu ce bordel et on cast en int
    idAvatar = int(dfAvatarID.loc[dfAvatarID["avatar_name"]== avatarName, "avatar_id"].to_string()[5:])
    df = df[df["avatar_id"] == idAvatar]
    return df["date"].min()

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

def reinitializeData(doInit = False):
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

# Function for the request creation
def generateRequest(nbReq, avatarGen, langGen, cityGen, dayGen, deviceGen):
    print("Generating requests...")
    tabReq = []
    dayGenForLoop = np.linspace(dayGen[0], dayGen[1], nbReq)
    dayGenForLoop = [round(x) for x in dayGenForLoop]
    try:
        for i in range(nbReq):
            tabReq.append([random.choice(avatarGen), random.choice(langGen), random.choice(cityGen), dayGenForLoop[i], random.choice(deviceGen)])
    except:
        pass
    np.save('./data/request.npy', tabReq)
    st.write("Request generated :")
    for i in range(0, math.floor(len(tabReq)), 3):
        try:
            st.markdown(str(tabReq[i])+str(tabReq[i+1])+str(tabReq[i+2]))
        except:
            try:
                st.markdown(str(tabReq[i])+str(tabReq[i+1]))
            except:
                st.markdown(str(tabReq[i]))

# Front for the request creation
def stGenRequest():
    if st.sidebar.checkbox("Random request generator"):
        avatarList = getAllAvatar()
        fromNb = st.number_input("From which avatar ?", min_value=1)
        toNb = st.number_input("To which avatar ?", min_value=1)
        avatarList = avatarList[fromNb-1:toNb]
        if st.button("Generate requests"):
            listReq = []
            for avatar in avatarList:
                listReq.append([avatar, random.choice(getAllLanguage()), 
                                random.choice(getAllCity()), int(random.choice(getAllDate())), 
                                random.choice([0,1])])
            st.write(listReq)
            np.save('./data/request.npy', listReq)

    else:
        col1, col2, col3, col4 = st.columns([1,1,1,2])
        with col3:
            #Number of request to generate
            nbReq = st.number_input("How many request ?", min_value=1)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            #option on avatar
            optAvatar = st.selectbox(
                'Avatar selection mode',
                ('All avatar randomly', 'Avatar selection'))
            if optAvatar != "All avatar randomly":
                avatarGen = st.multiselect(
                    'Generate request for which avatar ?',
                    getAllAvatar())
            else:
                avatarGen = list(getAllAvatar())

        with col2:
            #option on avatar
            optLangage = st.selectbox(
                'Language selection mode',
                ('All language randomly', 'Language selection'))
            if optLangage != "All language randomly":
                langGen = st.multiselect(
                    'Generate request for which language ?',
                    getAllLanguage())
            else:
                langGen = list(getAllLanguage())

        with col3:
            #option on avatar
            optCity = st.selectbox(
                'City selection mode',
                ('All city randomly', 'City selection'))
            if optCity != "All city randomly":
                cityGen = st.multiselect(
                    'Generate request for which city ?',getAllCity())
            else:
                cityGen = list(getAllCity())

        with col4:
            #option on avatar
            optDay = st.selectbox(
                'Day selection mode',
                ('Full interval', 'Day interval selection'))
            if optDay != "Full interval":
                dayFrom = int(st.selectbox(
                    "Start from ?",
                    getAllDate()))
                dayTo = int(st.selectbox(
                    'to ?',
                    getAllDate()))
                dayGen = [dayFrom, dayTo]
            else:
                dayGen = [44,0]

        with col5:
            #option on device, 1 is for mobile, 0 is for computer
            optMob = st.selectbox(
                'Device selection mode',
                ('Random', 'Mobile only', 'Computer only'))
            if optMob == "Mobile only":
                deviceGen = [1]
            elif optMob == "Computer only":
                deviceGen = [0]
            else:
                deviceGen = [0,1]
        if st.button("Generate requests"):
                generateRequest(nbReq, avatarGen, langGen, cityGen, dayGen, deviceGen)

def plotAvatarInfo(df):
    # import plotly.express as px
    

    # print(df)
    avID=pd.unique(df["avatar_id"])
    tabX= []
    tabY = []
    for avatID in avID:
        tabX.append(getAvatarName(avatID))
        tabY.append(getMinDayOfAvatar(getAvatarName(avatID)))

    # Horizontal Bar Plot
    fig = px.bar(x=tabX, y=tabY, labels=dict(x="Avatar name", y="Date"))
    st.plotly_chart(fig)

def plotPriceInfo(df):
    listHotel = list(pd.unique(df["hotel_id"]))
    listHotel.sort()
    choice = st.sidebar.selectbox("Tu veux plot quoi maggle ?", listHotel)
    dfHotel = df[df["hotel_id"] == choice]
    st.dataframe(df)

    col1, col2 = st.columns(2)

    with col1:
        byLang = dfHotel.groupby(["language"])["price"].agg('mean').reset_index().sort_values(by=['price'])
        st.plotly_chart(px.bar(byLang, x="language", y="price", labels=dict(x="langage", y="price")))

        byDate = dfHotel.groupby(["date"])["price"].agg('mean').reset_index().sort_values(by=['price'])
        st.plotly_chart(px.bar(byDate, x="date", y="price", labels=dict(x="date", y="price")))

    with col2:
        byStock = dfHotel.groupby(["stock"])["price"].agg('mean').reset_index().sort_values(by=['price'])
        st.plotly_chart(px.bar(byStock, x="stock", y="price", labels=dict(x="stock", y="price")))

        #TODO: link les avatar id a leur nom et display par nom
        byAvatar = dfHotel.groupby(["avatar_id"])["price"].agg('mean').reset_index().sort_values(by=['price'])
        idAvatar = pd.read_csv('./Data/AvatarNameAndID.csv')
        idAvatar = idAvatar[["avatar_name", "avatar_id"]]
        print(idAvatar)
        # dictTest = idAvatar.to_dict("avatar_id")
        # print(dictTest)
        # st.plotly_chart(px.bar(byAvatar, x="avatar_id", y="price", labels=dict(x="avatar_id", y="price")))

def stPlotting():
    st.header("Decide what you want to plot")
    df = pd.read_csv("./data/allData.csv")
    wtp = ["What to plot ?", "Avatar information", "Price"]
    choice = st.sidebar.selectbox("Tu veux plot quoi maggle ?", wtp)
    if choice == wtp[1]:
        plotAvatarInfo(df)
    if choice == wtp[2]:
        plotPriceInfo(df)


def stCreateAvatar():
    avatarList = getAllAvatar()[-1:]
    print(avatarList)
    id = int(re.findall('[0-9]+', avatarList[0])[0])
    nbAvToCreate = st.number_input("How many avatar do you want to create ?", min_value=1)
    st.write("It will create avatar from ", id+1, " to ", id+nbAvToCreate)
    if st.button("Generate requests"):
        for i in range(nbAvToCreate):
            idNb = id + i +1
            nameAvatar = "Avataricard" + str(idNb)
            makeRequest.createAvatar(nameAvatar)
        createAvatarIDcsv()
