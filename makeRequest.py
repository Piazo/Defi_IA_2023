import urllib.parse
import requests
import features
import numpy as np

domain = "51.91.251.0"
port = 3000
host = f"http://{domain}:{port}"
path = lambda x: urllib.parse.urljoin(host, x)
user_id = '18fcee0f-416b-4fd4-8fce-58c7d2030f43'


def createAvatar(nameAvatar):
    print("Creating the avatar ", nameAvatar)
    r = requests.post(path(f'avatars/{user_id}/{nameAvatar}'))
    features.addRequest("requests.post(path(f'avatars/{user_id}/{nameAvatar}'")
    features.addAvatar(nameAvatar)
    features.addResponseHistory(r)
    print("Avatar created")

# Usefull if we lose all the data, i hope it won't happen lol
def get_avatar():
    r = requests.get(path(f"avatars/{user_id}"))
    for avatar in r.json():
        print(avatar['id'], avatar['name'])


def pricingRequest(avatarName, language, city, date, mobile):
    print("Starting pricing request...")
    params = {
        "avatar_name": avatarName,
        "language": language,
        "city": city,
        "date": date,
        "mobile": mobile,}
    r = requests.get(path(f"pricing/{user_id}"), params=params)
    features.addRequest('requests.get(path(f"pricing/{user_id}"), params='+str(params)+')')
    features.addResponseHistory(r)
    print("Pricing request done !")


def main(create_avatar = False, doARequest = False):
    if create_avatar: createAvatar("Avataricard15")
    listReq = np.load("./data/request.npy")
    for req in listReq:
        avatar = req[0]
        language = req[1]
        city = req[2]
        date = req[3]
        mobile = req[4]
        if doARequest: pricingRequest(avatar, language, city, date, mobile)

if __name__=="__main__":
    main(False, True)

