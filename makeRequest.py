import urllib.parse
import requests

domain = "51.91.251.0"
port = 3000
host = f"http://{domain}:{port}"
path = lambda x: urllib.parse.urljoin(host, x)

user_id = '18fcee0f-416b-4fd4-8fce-58c7d2030f43'
name = 'first_avatar'
r = requests.post(path(f'avatars/{user_id}/{name}'))



