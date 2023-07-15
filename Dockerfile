FROM ubuntu:20.04

# Install Python, Vim, and necessary libraries
RUN apt-get update && \
    apt-get install -y software-properties-common wget gnupg2 lsb-release && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.6 python3-pip vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary dependencies for PostgreSQL and Rust
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev libpq-dev libclang-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PostgreSQL
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y  \
    postgresql-14 \
    postgresql-server-dev-14 \
    postgresql-plpython3-14 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Run as root
RUN mkdir /project && \
    chown postgres:postgres /project && \
    chown -R postgres:postgres /usr/share/postgresql/14/ && \
    chown -R postgres:postgres /usr/lib/postgresql/14/

# Switch to the postgres user and install Rust and pgrx
USER postgres
WORKDIR /project
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
RUN /bin/bash -c "source $HOME/.cargo/env && cargo install cargo-pgrx --version '0.9.7' --locked"
RUN /bin/bash -c "source $HOME/.cargo/env && cargo pgrx init --pg14 /usr/bin/pg_config && cargo pgrx new my_extension"

# Switch back to root to copy Cargo.toml.dev to the container
USER root
COPY ./Cargo.toml.dev /project/my_extension/Cargo.toml

# Switch back to postgres user
USER postgres

# Set environment variables for PostgreSQL
ENV PGDATA /var/lib/postgresql/data

# Initialize PostgreSQL data directory
RUN mkdir -p ${PGDATA} && chown -R postgres:postgres ${PGDATA} && \
    service postgresql start && \
    /usr/lib/postgresql/14/bin/initdb -D ${PGDATA}

# CMD statement to start PostgreSQL when the container starts
CMD service postgresql start && tail -F /var/log/postgresql/postgresql-14-main.log
