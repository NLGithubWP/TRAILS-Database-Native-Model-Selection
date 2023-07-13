FROM ubuntu:20.04

# Install Python
RUN apt-get update && \
    apt-get install -y software-properties-common \
    wget \
    gnupg2 \
    lsb-release && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.6 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PostgreSQL
RUN apt-get update && \
    apt-get install -y wget \
    gnupg2 \
    lsb-release && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y postgresql-14 \
    postgresql-plpython3-14 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for PostgreSQL
ENV PGDATA /var/lib/postgresql/data

# Initialize PostgreSQL data directory
RUN service postgresql start && su postgres -c "/usr/lib/postgresql/14/bin/initdb -D ${PGDATA}"

# CMD statement to start PostgreSQL when the container starts
CMD service postgresql start && tail -F /var/log/postgresql/postgresql-14-main.log
