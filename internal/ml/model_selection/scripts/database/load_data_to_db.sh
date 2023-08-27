#!/bin/bash

# Connection details
HOST="localhost"
PORT="28814"
USERNAME="postgres"
DBNAME="frappe"

# Create the database
echo "Creating database..."
createdb -h $HOST -p $PORT -U $USERNAME $DBNAME

# 1. Identify the number of columns
num_columns=$(awk 'NF > max { max = NF } END { print max }' /project/exp_data/data/structure_data/frappe/train.libsvm)

# 2. Create the table dynamically
create_table_cmd="CREATE TABLE frappe_train (id SERIAL PRIMARY KEY, label INTEGER"

for (( i=2; i<=$num_columns; i++ )); do
    create_table_cmd+=", col$(($i-1)) TEXT"
done
create_table_cmd+=");"

echo "Creating table..."
echo $create_table_cmd | psql -h $HOST -p $PORT -U $USERNAME -d $DBNAME

# 3. Transform the libsvm format to CSV
echo "Transforming data to CSV format..."
awk '{
    gsub(/:/, ","); # replaces : with ,
    print;
}' /project/exp_data/data/structure_data/frappe/train.libsvm > /project/exp_data/data/structure_data/frappe/train.csv

# 4. Import into PostgreSQL
columns="label"
for (( i=2; i<=$num_columns; i++ )); do
    columns+=", col$(($i-1))"
done

echo "Loading data into PostgreSQL..."
psql -h $HOST -p $PORT -U $USERNAME -d $DBNAME -c "\COPY frappe_train($columns) FROM '/project/exp_data/data/structure_data/frappe/train.csv' DELIMITER ' '"

echo "Data load complete."
