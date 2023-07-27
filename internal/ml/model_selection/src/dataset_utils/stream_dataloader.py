import queue
import threading
import requests
import torch


class StreamingDataLoader:
    def __init__(self, cache_svc_url):
        self.cache_svc_url = cache_svc_url
        self.data_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.fetch_data, daemon=True)
        self.thread.start()

    def fetch_data(self):
        while not self.stop_event.is_set():
            response = requests.get(f'{self.cache_svc_url}/')
            if response.status_code == 200:
                batch = response.json()

                # convert to tensor again
                id_tensor = torch.LongTensor(batch['id'])
                value_tensor = torch.FloatTensor(batch['value'])
                y_tensor = torch.FloatTensor(batch['y'])
                data_tensor = {'id': id_tensor, 'value': value_tensor, 'y': y_tensor}
                print(f"put to data queue: {data_tensor}")
                self.data_queue.put(data_tensor)
            else:
                print(response.json())
                time.sleep(5)

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_queue.empty() and not self.thread.is_alive():
            raise StopIteration
        else:
            return self.data_queue.get(block=True)

    def __len__(self):
        return self.data_queue.qsize()

    def stop(self):
        self.stop_event.set()
        self.thread.join()


if __name__ == "__main__":

    import requests
    url = 'http://localhost:8093/'
    columns = ['col1', 'col2', 'col3', 'label']
    response = requests.post(url, json={'columns': columns})
    print(response.json())

    stream = StreamingDataLoader(cache_svc_url="http://localhost:8093")
    import time
    for batch_idx, batch in enumerate(stream):
        print(batch_idx, batch)
        time.sleep(1)
