from sanic import Sanic

from sanic.views import HTTPMethodView
from sanic.response import json as sjs
import json

app = Sanic("FastAutoNAS")


class MockETCD:

    def __init__(self):
        # model - score pair.
        self.mode_score_pair_ls = []
        # a list of new model_id_ls
        self.model_id_ls = []

        self.status = False

    # worker append score result to list
    def append_score(self, score):
        self.mode_score_pair_ls.append(score)
        return self._get_model()

    # worker then get a new model
    def _get_model(self) -> dict:
        if len(self.model_id_ls) > 0:
            model = self.model_id_ls[0]
            self.model_id_ls = self.model_id_ls[1:]
            return {"status": self.status, "model_id": model}
        else:
            return {"status": self.status, "model_id": -1}

    # search strategy read score
    def get_score(self):
        if len(self.mode_score_pair_ls) > 0:
            model_score = self.mode_score_pair_ls[0]
            self.mode_score_pair_ls = self.mode_score_pair_ls[1:]
            return model_score
        else:
            return -1

    # search strategy append model to mode id list
    def append_model(self, model):
        self.model_id_ls.append(model)


@app.route('/return_score', methods=['GET'])
async def return_score(request):
    global metcd
    data = json.loads(request.json)
    model_id = data["model_id"]
    score = data["score"]
    new_model_id = metcd.append_score((model_id, score))
    return sjs({'model_id': new_model_id})


@app.route('/select_model', methods=['GET'])
async def hello(request):
    global metcd
    data = json.loads(request.json)
    model_id = data["model_id"]
    score = data["score"]
    new_model_id = metcd.append_score((model_id, score))
    return sjs({'model_id': new_model_id})


@app.route('/select_model', methods=['GET'])
async def hello(request):
    global metcd
    data = json.loads(request.json)
    model_id = data["model_id"]
    score = data["score"]
    new_model_id = metcd.append_score((model_id, score))
    return sjs({'model_id': new_model_id})


@app.route('/select_model', methods=['GET'])
async def hello(request):
    global metcd
    data = json.loads(request.json)
    model_id = data["model_id"]
    score = data["score"]
    new_model_id = metcd.append_score((model_id, score))
    return sjs({'model_id': new_model_id})


if __name__ == '__main__':
    metcd = MockETCD()

    run(host="0.0.0.0", port=8000, debug=True)


