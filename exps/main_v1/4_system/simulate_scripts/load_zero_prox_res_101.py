
from pprint import pprint
import os, pickle
from tqdm import tqdm


d = '../trails/data/nasbench1/proxies'
runs = []
processed = set()

for f in tqdm(os.listdir(d)):
    pf = open(os.path.join(d,f),'rb')
    while 1:
        try:
            p = pickle.load(pf)
            if p['hash'] in processed:
                continue
            processed.add(p['hash'])
            runs.append(p)
        except EOFError:
            break
    pf.close()

print(len(runs))
pprint(runs[0])
