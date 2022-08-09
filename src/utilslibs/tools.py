import json


def read_json(file_name):
    with open(file_name, 'r') as readfile:
        data = json.load(readfile)
    return data


def write_json(file_name, data):
    with open(file_name, 'w') as outfile:
        outfile.write(json.dumps(data))




