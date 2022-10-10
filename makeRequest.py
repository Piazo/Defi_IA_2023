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

# def get_avatar():
#     r = requests.get(path(f"avatars/{user_id}"))
#     for avatar in r.json():
#         print(avatar['id'], avatar['name'])


def pricingRequest(avatarName, language, city, date, mobile):
    params = {
        "avatar_name": avatarName,
        "language": language,
        "city": city,
        "date": date,
        "mobile": mobile,}
    r = requests.get(path(f"pricing/{user_id}"), params=params)
    features.addRequest('requests.get(path(f"pricing/{user_id}"), params='+str(params)+')')
    features.addResponseHistory(r)
    return r

def saveResp():
    r = pricingRequest("Avataricard001", "french", "paris", 44, 0)
    df = pd.read_json(r)
    print(df)
    df.to_csv('./data/testtodf.csv')


# def createDataFromRequest(requests):
#     pricing_requests = []
#     for r in requests:
#         pricing_requests.append(
#             pd.DataFrame(r.json()['prices']).assign(**r.json()['request'])
#         )

#     pricing_requests = pd.concat(pricing_requests)
#     pricing_requests.head()


# def contatenation():
#     pass

def main(create_avatar = False, doARequest = False):
    if create_avatar: createAvatar("Avataricard001")
    if doARequest: saveResp()

main(False, True)
