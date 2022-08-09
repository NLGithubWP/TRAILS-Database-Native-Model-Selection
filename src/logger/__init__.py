

import logging
import os

if not os.path.exists("./Logs"):
    os.makedirs("./Logs")

logger = logging.getLogger(__name__)

if os.environ.get("log_file_name") == None:
    log_name = "./Logs/test.log"
else:
    log_name = "./Logs/" + os.environ.get("log_file_name")


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S',
                    filename=log_name, filemode='w')
