

docker build -t trails .

docker run -d --name trails \
  -v $(pwd)/TRAILS:/TRAILS \
  -v $(pwd)/postgresdata/data:/var/lib/postgresql/data \
  -e POSTGRES_USER=trails \
  -e POSTGRES_PASSWORD=trails \
  trails

docker run -d --name trails \
  -v $(pwd)/TRAILS:/project/TRAILS \
  trails


su postgres
psql


# psql test
CREATE EXTENSION plpython3u;

CREATE FUNCTION py_version() RETURNS text AS $$
import sys
return sys.version
$$ LANGUAGE plpython3u;

SELECT py_version();


CREATE OR REPLACE FUNCTION test_numpy()
  RETURNS text
LANGUAGE plpython3u
AS $$
import numpy
import torch
import sklearn
import torchvision
import tqdm
print("asdf")
return str(numpy.__version__) + " torch: " + str(torch.__version__)
$$;

SELECT test_numpy();

CREATE EXTENSION my_extension;
SELECT hello_my_extension();

# this is on mac locally
conda activate firmest38
export PYTHON_SYS_EXECUTABLE=/Users/kevin/opt/anaconda3/envs/firmest38/bin/python
export DYLD_LIBRARY_PATH=/Users/kevin/opt/anaconda3/envs/firmest38/lib/:$DYLD_LIBRARY_PATH
cargo run --features python

# this is in docker image already
cargo install --locked cargo-pgrx
cargo pgrx init --pg14 /usr/bin/pg_config
cargo pgrx new my_extension
cargo pgrx run

# on host after sync.sh
chmod -R 777 internal/pg_extension

# on container

## 1. ensure the pyhton modules called in the UDF can find
export PYTHONPATH=$PYTHONPATH:/project/TRAILS/internal/ml/model_selec

## 2. compile udf
cargo pgrx run
CREATE EXTENSION pg_extension;

## 3. run it
SELECT "hello pg"('your_task_here');
