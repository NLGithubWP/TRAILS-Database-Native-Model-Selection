#!/bin/bash

# Configurations
DATA_PATH="/project/exp_data/data/structure_data/frappe"   # Edit this path as needed
DB_NAME="frappe"   # Edit the database name as needed

# Connection details
HOST="localhost"
PORT="28814"
USERNAME="postgres"
DBNAME=$DB_NAME

rm ${DATA_PATH}/train.csv

# Create the database
echo "Creating database..."
createdb -h $HOST -p $PORT -U $USERNAME $DBNAME

# 1. Identify the number of columns
num_columns=$(awk 'NF > max { max = NF } END { print max }' ${DATA_PATH}/train.libsvm)

# 2. Create the table dynamically
create_table_cmd="CREATE TABLE ${DB_NAME}_train (id SERIAL PRIMARY KEY, label INTEGER"

for (( i=2; i<=$num_columns; i++ )); do
    create_table_cmd+=", col$(($i-1)) TEXT"
done
create_table_cmd+=");"

echo "Creating table..."
echo $create_table_cmd | psql -h $HOST -p $PORT -U $USERNAME -d $DBNAME

# 3. Transform the libsvm format to CSV
echo "Transforming data to CSV format..."

awk '{
    for (i = 1; i <= NF; i++) {
        printf "%s", $i;  # print each field as-is
        if (i < NF) {
            printf " ";  # if its not the last field, print a space
        }
    }
    printf "\n";  # end of line
}' ${DATA_PATH}/train.libsvm > ${DATA_PATH}/train.csv

# 4. Import into PostgreSQL
columns="label"
for (( i=2; i<=$num_columns; i++ )); do
    columns+=", col$(($i-1))"
done

echo "Loading data into PostgreSQL..."
psql -h $HOST -p $PORT -U $USERNAME -d $DBNAME -c "\COPY ${DB_NAME}_train($columns) FROM '${DATA_PATH}/train.csv' DELIMITER ' '"

echo "Data load complete."
