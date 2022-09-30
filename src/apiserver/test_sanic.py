import os

from sanic import Sanic
from sanic.response import text

os.environ.setdefault("log_file_name", "123123")
from logger import logger

app = Sanic('logging_example')


@app.route('/')
async def test(request):
    logger.info('Here is your log!!')
    return text('Hello World!')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=False, access_log=False)


