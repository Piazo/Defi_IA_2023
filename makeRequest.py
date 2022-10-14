import urllib.parse
import requests
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

"""
tab_request = [
                ["Avataricard01", 'austrian', 'amsterdam', 44, 1],
                ["Avataricard01", 'austrian', 'copenhagen', 44, 0],

                ["Avataricard01", 'bulgarian', 'amsterdam', 44, 1],
                ["Avataricard01", 'bulgarian', 'copenhagen', 44, 0],

                ["Avataricard01", 'croatian', 'madrid', 44, 1],
                ["Avataricard01", 'croatian', 'paris', 44, 0],

                ["Avataricard01", 'cypriot', 'madrid', 44, 0],
                ["Avataricard01", 'cypriot', 'paris', 44, 1],

                ["Avataricard01", 'danish', 'rome', 20, 1],
                ["Avataricard01", 'danish', 'sofia', 20, 0],

                ["Avataricard01", 'dutch', 'rome', 20, 0],
                ["Avataricard01", 'dutch', 'sofia', 20, 1],

                ["Avataricard01", 'estonian', 'valletta', 20, 0],
                ["Avataricard01", 'estonian', 'vienna', 20, 1],

                ["Avataricard01", 'finnish', 'valletta', 20, 0],
                ["Avataricard01", 'finnish', 'vienna', 20, 1],

                ["Avataricard01", 'german', 'vilnius', 20, 0],
                ["Avataricard01", 'german', 'sofia', 20, 1],

                ["Avataricard01", 'irish', 'amsterdam', 20, 0],
                ["Avataricard01", 'irish', 'copenhagen', 20, 1],

                ["Avataricard01", 'italian', 'madrid', 20, 1],
                ["Avataricard01", 'italian', 'paris', 20, 0],

                ["Avataricard01", 'latvian', 'rome', 20, 0],
                ["Avataricard01", 'latvian', 'paris', 20, 1],

                ["Avataricard01", 'lithuanian', 'valletta', 20, 0],
                ["Avataricard01", 'lithuanian', 'vienna', 20, 1],

                ["Avataricard01", 'luxembourgish', 'vilnius', 20, 0],
                ["Avataricard01", 'luxembourgish', 'sofia', 20, 1],

                ["Avataricard01", 'maltese', 'amsterdam', 12, 0],
                ["Avataricard01", 'maltese', 'copenhagen', 12, 1],

                ["Avataricard01", 'polish', 'madrid', 12, 0],
                ["Avataricard01", 'polish', 'paris', 12, 1],

                ["Avataricard01", 'portuguese', 'rome', 12, 0],
                ["Avataricard01", 'portuguese', 'valletta', 12, 1],

                ["Avataricard01", 'romanian', 'vienna', 12, 0],
                ["Avataricard01", 'romanian', 'vilnius', 12, 1],

                ["Avataricard01", 'slovakian', 'amsterdam', 12, 0],
                ["Avataricard01", 'slovakian', 'copenhagen', 12, 1],

                ["Avataricard01", 'slovene', 'madrid', 12, 0],
                ["Avataricard01", 'slovene', 'paris', 12, 1],

                ["Avataricard01", 'spanish', 'rome', 12, 0],
                ["Avataricard01", 'spanish', 'sofia', 12, 1],
            ]
"""

def main(create_avatar = False, doARequest = False):
    if create_avatar: createAvatar("Avataricard05")
    avatar = "Avataricard02"
    language = "hungarian"
    city = "rome"
    date = 2 
    mobile = 0
    if doARequest: pricingRequest(avatar, language, city, date, mobile)

if __name__=="__main__":
    main(False, False)

