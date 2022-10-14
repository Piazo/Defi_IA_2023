import streamlit as st
import features
import numpy as np
import random
st. set_page_config(layout="wide")


# Function for the request creation
def generateRequest(nbReq, avatarGen, langGen, cityGen, dayGen, deviceGen):
    tabReq = []
    print(dayGen)
    dayGenForLoop = np.linspace(dayGen[0], dayGen[1], nbReq)
    dayGenForLoop = [round(x) for x in dayGenForLoop]
    for i in range(nbReq):
        tabReq.append([random.choice(avatarGen), random.choice(langGen), random.choice(cityGen), dayGenForLoop[i], random.choice(deviceGen)])
    print(tabReq)


# dayGen = np.linspace(dayFrom, dayTo, dayFrom-dayTo+1)

# Front for the request creation
def createReq():
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
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
                features.getAllAvatar())
        else:
            avatarGen = list(features.getAllAvatar())

    with col2:
        #option on avatar
        optLangage = st.selectbox(
            'Language selection mode',
            ('All language randomly', 'Language selection'))
        if optLangage != "All language randomly":
            langGen = st.multiselect(
                'Generate request for which language ?',
                features.getAllLanguage())
        else:
            langGen = list(features.getAllLanguage())

    with col3:
        #option on avatar
        optCity = st.selectbox(
            'City selection mode',
            ('All city randomly', 'City selection'))
        if optCity != "All city randomly":
            cityGen = st.multiselect(
                'Generate request for which city ?',
                features.getAllCity())
        else:
            cityGen = list(features.getAllCity())

    with col4:
        #option on avatar
        optDay = st.selectbox(
            'Day selection mode',
            ('Full interval', 'Day interval selection'))
        if optDay != "Full interval":
            dayFrom = int(st.selectbox(
                "Start from ?",
                features.getAllDate()))
            dayTo = int(st.selectbox(
                'to ?',
                features.getAllDate()))
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
    
    st.button("Generate requests", on_click=generateRequest(nbReq, avatarGen, langGen, cityGen, dayGen, deviceGen))

if __name__=="__main__":
    createReq()