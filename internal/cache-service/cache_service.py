import time
import threading
import queue
import psycopg2
from typing import Any, List, Dict, Tuple
from sanic import Sanic
from sanic.response import json
import calendar
import os
import logging

log_logger_folder_name = "log_cache_service"
if not os.path.exists(f"./{log_logger_folder_name}"):
    os.makedirs(f"./{log_logger_folder_name}")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S',
                    filename=f"./{log_logger_folder_name}/log_{str(calendar.timegm(time.gmtime()))}", filemode='w')

USER = "postgres"
HOST = "127.0.0.1"
PORT = "28814"
DB_NAME = "pg_extension"
TABLE = "dummy"
CACHE_SIZE = 10


class CacheService:
    def __init__(self, database: str, table: str, columns: List, max_size: int = CACHE_SIZE):
        """
        database: database to use
        table: which table
        columns: selected cols
        max_size: max batches to cache
        """
        self.database = database
        self.table = table
        self.columns = columns
        self.queue = queue.Queue(maxsize=max_size)
        self.thread = threading.Thread(target=self.fetch_data, daemon=True)
        self.thread.start()

    def decode_libsvm(self, columns):
        map_func = lambda pair: (int(pair[0]), float(pair[1]))
        id, value = zip(*map(lambda col: map_func(col.split(':')), columns[:-1]))
        sample = {'id': list(id),
                  'value': list(value),
                  'y': float(columns[-1])}
        return sample

    def pre_processing(self, mini_batch_data: List[Tuple]):
        """
        mini_batch_data: [('123:123', '123:123', '123:123', '0')
        """
        sample_lines = len(mini_batch_data)
        nfields = len(mini_batch_data[0]) - 1
        feat_id = []
        feat_value = []
        y = []

        for i in range(sample_lines):
            row_value = mini_batch_data[i]
            sample = self.decode_libsvm(list(row_value))
            feat_id.append(sample['id'])
            feat_value.append(sample['value'])
            y.append(sample['y'])
        return {'id': feat_id, 'value': feat_value, 'y': y}

    def fetch_data(self):
        with psycopg2.connect(database=self.database, user=USER, host=HOST, port=PORT) as conn:
            while True:
                try:
                    # fetch and preprocess data from PostgreSQL
                    batch = self.fetch_and_preprocess(conn)
                    # block until a free slot is available
                    self.queue.put(batch)
                    time.sleep(0.1)
                except psycopg2.OperationalError:
                    logger.exception("Lost connection to the database, trying to reconnect...")
                    time.sleep(5)  # wait before trying to establish a new connection
                    conn = psycopg2.connect(database=self.database, user=USER, host=HOST, port=PORT)

    def fetch_and_preprocess(self, conn):
        begin_time = time.time()
        cur = conn.cursor()
        # Assuming you want to get the latest 100 rows
        columns_str = ', '.join(self.columns)
        cur.execute(f"SELECT {columns_str} FROM {self.table} LIMIT 10")
        rows = cur.fetchall()
        batch = self.pre_processing(rows)

        logger.info(f"Data is fetched {time.time() - begin_time}")

        return batch

    def get(self):
        return self.queue.get()

    def is_empty(self):
        return self.queue.empty()


app = Sanic("CacheServiceApp")


# todo: here we don't care abou the concurrency, so only one CacheService for one task usage.
# start the server
@app.route("/", methods=["POST"])
async def start_service(request):
    try:
        columns = request.json.get('columns')
        if columns is None:
            return json({"error": "No columns specified"}, status=400)

        print(f"columns are {columns}")

        if not hasattr(app.ctx, 'cache_service'):
            app.ctx.cache_service = CacheService(DB_NAME, TABLE, columns, CACHE_SIZE)

        return json("OK")
    except Exception as e:
        return json({"error": str(e)}, status=500)


@app.route("/", methods=["GET"])
async def serve_get_request(request):
    if not hasattr(app.ctx, 'cache_service'):
        print(f"cache_service not start yet")
        return json({"error": "cache_service not start yet"}, status=404)

    data = app.ctx.cache_service.get()
    print(f"return data {data}")
    if data is None:
        return json({"error": "No data available"}, status=404)
    else:
        return json(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8093)
