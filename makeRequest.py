import urllib.parse
import requests
import pandas as pd
import features


domain = "51.91.251.0"
port = 3000
host = f"http://{domain}:{port}"
path = lambda x: urllib.parse.urljoin(host, x)
user_id = '18fcee0f-416b-4fd4-8fce-58c7d2030f43'


def createAvatar(nameAvatar):
    r = requests.post(path(f'avatars/{user_id}/{nameAvatar}'))
    features.addRequest("requests.post(path(f'avatars/{user_id}/{nameAvatar}'")
    features.addAvatar(nameAvatar)
    features.addResponseHistory(r)
    print(r)

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
    if create_avatar: createAvatar("Avataricard01")
    if doARequest: pricingRequest("Avataricard01", 'french', 'paris', 44, 0)

main(False, True)
