

import logging
import os
import calendar
import time

if not os.path.exists("./Logs"):
    os.makedirs("./Logs")

logger = logging.getLogger(__name__)

gmt = time.gmtime()
ts = calendar.timegm(gmt)

log_name = "./Logs/fast_auto_nas_log_"+str(ts)+".log"

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S',
                    filename=log_name, filemode='w')
