import os
import psutil
import gpustat
import threading
import time
from src.tools.io_tools import write_json


def print_cpu_gpu_usage(interval=1, output_file="path_to_folder"):
    def print_usage():
        # Get current process
        process = psutil.Process(os.getpid())

        # Create an empty dictionary to store metrics
        metrics = {'cpu_usage': [], 'memory_usage': [], 'gpu_usage': []}

        while True:
            # Get CPU usage for current process
            cpu_percent = process.cpu_percent()
            metrics['cpu_usage'].append(cpu_percent)

            # Get memory usage for current process
            memory_info = process.memory_info()
            # bytes -> MB
            metrics['memory_usage'].append(memory_info.rss / (1024 ** 2))

            # Get GPU usage
            gpu_stats = gpustat.GPUStatCollection.new_query()
            for gpu in gpu_stats:
                # here in MB
                metrics['gpu_usage'].append((gpu.index, gpu.utilization, gpu.memory_used))

            # If it's time to write metrics to a file, do so
            if len(metrics['cpu_usage']) % 4 == 0:
                write_json(output_file, metrics)

            time.sleep(interval)

    thread = threading.Thread(target=print_usage)
    thread.start()
