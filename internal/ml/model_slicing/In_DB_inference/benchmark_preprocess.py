

import time
import torch
import time
from typing import Any, List, Dict, Tuple

mini_batch = [
    ('4801', '0', '2:1', '4656:1', '5042:1', '5051:1', '5054:1', '5055:1', '5058:1', '5061:1', '5070:1', '5150:1'),
]

mini_batch = mini_batch * 100000
print(len(mini_batch))
begin = time.time()
mini_batch_raw = [
    [int(item.split(':')[0]) for item in sublist[2:]]
    for sublist in mini_batch]
print(time.time() - begin)





