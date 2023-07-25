

# Build and run the container

```bash
docker build -t trails .

docker run -d --name trails \
  -v $(pwd)/TRAILS:/project/TRAILS \
  -v $(pwd)/exp_data:/project/exp_data \
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

# MAC locally
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

# Develop

## Create dummy data

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

## 1. Compile

In shell

```bash
cd internal/pg_extension/
cargo clean
rm -r /usr/lib/postgresql/14/lib/pg_extension.so
cargo pgrx run
```

In SQL

```sql
DROP EXTENSION IF EXISTS pg_extension;
CREATE EXTENSION pg_extension;
```

## 2. Edit the config file

Update the `nfield` in the `config.ini` file, it is == number of columns used.

E.g, `ARRAY['col1', 'col2', 'col3', 'label']`  => `nfield` = 3

## 3. Run it

```sql
CREATE EXTENSION pg_extension;

# test if the UDF is there or not
SELECT *  FROM pg_proc  WHERE proname = 'coordinator';
SELECT coordinator('0.0211558125', '5.122203075885773', '3330', false, '/project/TRAILS/internal/ml/model_selection/config.ini');

# this is database name, columns used, time budget, batch size, and config file
CALL model_selection_sp('dummy', ARRAY['col1', 'col2', 'col3', 'label'], '100', 32, '/project/TRAILS/internal/ml/model_selection/config.ini');
                
```

# Container log

Each line in your output represents a different process that is currently running on your PostgreSQL server. Here's what each one is doing:

1.  `/bin/sh -c service postgresql start && tail -F /var/log/postgresql/postgresq` : This is the command that was used to start your PostgreSQL server. It also includes a command to continuously display new entries from the PostgreSQL log file.


2.  `/usr/lib/postgresql/14/bin/postgres -D /var/lib/postgresql/14/main -c config` : This is the main PostgreSQL process. All other PostgreSQL processes are children of this process.


3.  `postgres: 14/main: checkpointer` : The checkpointer process is responsible for making sure data changes get saved to disk regularly. This is important for database recovery in case of a crash.


4.  `postgres: 14/main: background writer` : The background writer process is responsible for writing buffers to disk when they become dirty. This reduces the amount of work that needs to be done when a buffer is reused.


5.  `postgres: 14/main: walwriter` : The walwriter process writes transaction logs (Write-Ahead Logs or WAL) to disk. This is also important for database recovery and replication.


6.  `postgres: 14/main: autovacuum launcher` : The autovacuum launcher process starts autovacuum worker processes as needed. These processes automatically clean up and optimize the database.


7.  `postgres: 14/main: stats collector` : The stats collector process collects statistics about the server's activity. This information can be viewed using the `pg_stat` family of system views.


8.  `postgres: 14/main: logical replication launcher` : The logical replication launcher manages the worker processes that perform logical replication, copying data changes to other databases.


9.  `tail -F /var/log/postgresql/postgresql-14-main.log` : This process is displaying the end of the PostgreSQL log file and updating as more entries are added.


10.  `bash` : These are shell sessions, likely interactive ones you've started.


11.  `/usr/lib/postgresql/14/bin/psql -h localhost -p 28814 pg_extension` : These are instances of the psql command line interface, connected to your database.


12.  `postgres: postgres pg_extension 127.0.0.1(52236) CALL` : This is your currently running stored procedure.


13.  `ps aux` : This is the command you ran to display the list of processes.

Each process is part of the PostgreSQL database system and helps it to run efficiently and robustly.






