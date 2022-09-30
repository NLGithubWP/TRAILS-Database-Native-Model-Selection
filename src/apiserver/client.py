

import requests


def send_post_request(url, data):
    r = requests.post(url, json=data)
    print(r.text)


def send_get_request(url, data):
    r = requests.get(url, json=data)
    return r


if __name__ == "__main__":
    send_get_request("http://localhost:8000/get_model", "hellow")
    requests.get("http://0.0.0.0:8002/get_model", json="hellow")





