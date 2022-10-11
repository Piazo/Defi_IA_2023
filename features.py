import numpy as np

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

    # print(np.load('./data/avatar.npy'))

main(False)