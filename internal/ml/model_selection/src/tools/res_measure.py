import os
import psutil
import gpustat
import threading
import time
from src.tools.io_tools import write_json


def print_cpu_gpu_usage(interval=1, output_file="path_to_folder", stop_event=None):
    def print_usage():
        # Get current process
        main_process = psutil.Process(os.getpid())

        # Create an empty dictionary to store metrics
        metrics = {'cpu_usage': [], 'memory_usage': [], 'gpu_usage': []}

        while not stop_event.is_set():
            cpu_percent = 0
            mem_usage_mb = 0
            for process in main_process.children(recursive=True):  # Include all child processes
                cpu_percent += process.cpu_percent()
                mem_usage_mb += process.memory_info().rss / (1024 ** 2)

            metrics['cpu_usage'].append(cpu_percent)
            metrics['memory_usage'].append(mem_usage_mb)

            # Get GPU usage
            gpu_stats = gpustat.GPUStatCollection.new_query()
            for gpu in gpu_stats:
                # here in MB
                metrics['gpu_usage'].append((gpu.index, gpu.utilization, gpu.memory_used))
                # print(f"GPU {gpu.index} usage: {gpu.utilization}% memory: {gpu.memory_used}MB")

            # If it's time to write metrics to a file, do so
            if len(metrics['cpu_usage']) % 4 == 0:
                write_json(output_file, metrics)

            time.sleep(interval)

    stop_event = stop_event or threading.Event()
    thread = threading.Thread(target=print_usage)
    thread.start()
    return stop_event, thread
