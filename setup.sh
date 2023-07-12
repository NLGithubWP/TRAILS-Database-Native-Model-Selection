#!/bin/bash
set -e

#echo "shared_preload_libraries = 'plpython3u'" >> "$PGDATA/postgresql.conf"
#
#psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
#    CREATE EXTENSION plpython3u;
#EOSQL
