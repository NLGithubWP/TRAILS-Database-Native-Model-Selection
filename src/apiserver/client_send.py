import time

import requests


while True:
    # 172.28.176.55
    print(requests.get("http://0.0.0.0:8002/").json())
    time.sleep(3)
