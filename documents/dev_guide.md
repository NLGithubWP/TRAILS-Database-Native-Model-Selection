

# Build and run the container

```bash
docker build -t trails .

docker run -d --name trails \
  -v $(pwd)/TRAILS:/project/TRAILS \
  trails
```

# Test the pg-extension works

```sql
su postgres
psql

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
SELECT hello_my_extension();=
```

# mac locally
```bash
conda activate firmest38
export PYTHON_SYS_EXECUTABLE=/Users/kevin/opt/anaconda3/envs/firmest38/bin/python
export DYLD_LIBRARY_PATH=/Users/kevin/opt/anaconda3/envs/firmest38/lib/:$DYLD_LIBRARY_PATH
cargo run --features python
```

# this is in docker image already
```bash
cargo install --locked cargo-pgrx
cargo pgrx init --pg14 /usr/bin/pg_config
cargo pgrx new my_extension
cargo pgrx run
```

# On host after sync.sh
```bash
chmod -R 777 internal/pg_extension
chmod -R 777 TRAILS
```

# On container

## 1. Ensure the pyhton modules called in the UDF can find
```bash
export PYTHONPATH=$PYTHONPATH:/project/TRAILS/internal/ml/model_selection
```

## 2. Compile

In shell

```bash
cd internal/pg_extension/
cargo clean
rm -r /usr/lib/postgresql/14/lib/pg_extension.so
cargo pgrx run
```

In sql

```sql
DROP EXTENSION IF EXISTS pg_extension;
CREATE EXTENSION pg_extension;
```

## 3. Run it
```sql
SELECT filtering_phase('your_task_here');
SELECT system_profiling('your_task_here');
SELECT refinement_phase('your_task_here');
SELECT coordinator('your_task_here');
SELECT model_selection('your_dataset', ARRAY['col1', 'col2'], 2);
```

# Dev

Update code locally

In docker container, Run 

```bash
cargo pgrx run
```

Then run

```sql
CREATE EXTENSION pg_extension;

SELECT filtering_phase('your_task_here');
SELECT model_selection('your_dataset', ARRAY['col1', 'col2'], 2);
CALL model_selection_sp('dummy', ARRAY['col1', 'col2', 'col3', 'label'], 2);
```

# Data generate

```sql
CREATE TABLE dummy (
    id SERIAL PRIMARY KEY,
    col1 TEXT,
    col2 TEXT,
    col3 TEXT,
    col4 TEXT,
    col5 TEXT,
    col6 TEXT,
    col7 TEXT,
    col8 TEXT,
    col9 TEXT,
    label TEXT
);


INSERT INTO dummy (col1, col2, col3, col4, col5, col6, col7, col8, col9, label)
SELECT '123:123', '123:123', '123:123', '123:123', '123:123', '123:123', '123:123', '123:123', '123:123',
       CASE 
            WHEN random() < 0.5 THEN '0'
            ELSE '1'
       END
FROM generate_series(1,5000);


select * from dummy limit 10;

```









