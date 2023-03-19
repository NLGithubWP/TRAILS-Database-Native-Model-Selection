import threading
import time

from sanic import Sanic, response


class Singleton(object):
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance


class MyTimeManager(Singleton):
    _mytime = 0

    def updateTime(self):
        self._mytime = int(round(time.time() * 1000))
        print("updateTime", self._mytime)
        threading.Timer(1, self.updateTime).start()\

    @property
    def mytime(self):
        return self._mytime


app = Sanic(__name__)
MyTimeManager().updateTime()


@app.route("/")
async def test(request):
    return response.json({"test": True, "mytime": MyTimeManager().mytime})


if __name__ == '__main__':
    run(host="0.0.0.0", port=8080, workers=2)