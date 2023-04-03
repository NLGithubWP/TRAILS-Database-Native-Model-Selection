import os
import traceback

from sanic.request import Request
from sanic import Sanic
from sanic import response
import json

os.environ.setdefault("log_file_name", "123123")

app = Sanic('logging_example')


@app.route('/profile', methods=['POST'])
async def example(request: Request):
    try:
        data = request.form
        datas = json.loads(data["data"][0])
        print(datas)
    except:
        print(traceback)
    # process the data
    return response.json({'success': True})



@app.route('/start_refinement', methods=['GET'])
async def example(request: Request):
    print("start refinement")
    return response.json({'success': True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=False, access_log=False)


