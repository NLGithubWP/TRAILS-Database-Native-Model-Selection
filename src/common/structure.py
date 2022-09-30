import json


class ModelEvaData:
    """
    Eva worker send score to search strategy
    """

    def __init__(self, model_id: str = None, model_score: dict = None):
        if model_score is None:
            model_score = {}
        self.model_id = model_id
        self.model_score = model_score

    def serialize_model(self) -> str:
        data = {"model_id": self.model_id,
                "model_score": self.model_score}
        return json.dumps(data)

    @classmethod
    def deserialize(cls, data_str: str):
        data = json.loads(data_str)
        res = cls(
            data["model_id"],
            data["model_score"])
        return res


class ModelAcquireData:
    """
    Eva worker get model from search strategy
    """

    def __init__(self, model_id: str, model_encoding: str, is_last: bool = False):
        self.is_last = is_last
        self.model_id = model_id
        self.model_encoding = model_encoding

    def serialize_model(self) -> str:
        data = {"is_last": self.is_last,
                "model_id": self.model_id,
                "model_encoding": self.model_encoding}
        return json.dumps(data)

    @classmethod
    def deserialize(cls, data_str: str):
        data = json.loads(data_str)
        res = cls(
            data["model_id"],
            data["model_encoding"],
            data["is_last"])
        return res


if __name__ == "__main__":
    data = ModelEvaData("1", {"a": 1, "b": 2})
    data_str = data.serialize_model()
    res = ModelEvaData.deserialize(data_str)
    print(res)
