
import logging


class SingletonLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            # Pass the directory to _initialize_logger
            cls._instance._initialize_logger(kwargs.get('log_file_name', './'))  # initialize the logger once
        return cls._instance

    def _initialize_logger(self, log_file_name: str):
        self.logger = logging.getLogger('logger')

        # Use a FileHandler to output to a file
        handler = logging.FileHandler(f'{log_file_name}.txt')

        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

import os
if os.environ.get("log_logger_folder_name") == None:
    log_logger_folder_name = "Logs"
else:
    log_logger_folder_name = os.environ.get("log_logger_folder_name")

if not os.path.exists(f"./{log_logger_folder_name}"):
    os.makedirs(f"./{log_logger_folder_name}")

if os.environ.get("log_file_name") == None:
    log_name = f"./{log_logger_folder_name}/test.log"
else:
    log_name = f"./{log_logger_folder_name}/" + os.environ.get("log_file_name")


# Usage
logger = SingletonLogger(log_file_name=log_name).logger


