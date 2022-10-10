from ctypes.wintypes import BOOL
import urllib.parse
import requests
import pandas as pd
import features

# print(features.cityList, features.languageList)

domain = "51.91.251.0"
port = 3000
host = f"http://{domain}:{port}"
path = lambda x: urllib.parse.urljoin(host, x)

def create_avatar(nameAvatar):
    user_id = '18fcee0f-416b-4fd4-8fce-58c7d2030f43'
    name = nameAvatar
    r = requests.post(path(f'avatars/{user_id}/{name}'))







def createDataFromRequest(requests):
    pricing_requests = []
    for r in requests:
        pricing_requests.append(
            pd.DataFrame(r.json()['prices']).assign(**r.json()['request'])
        )

    pricing_requests = pd.concat(pricing_requests)
    pricing_requests.head()


def contatenation():
    pass

def main(create_avatar = False):
    if create_avatar: create_avatar("test_avatar")

main()
