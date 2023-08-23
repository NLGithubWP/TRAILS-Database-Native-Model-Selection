FROM ubuntu:20.04

# Install Python, Vim, and necessary libraries
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y software-properties-common wget gnupg2 lsb-release git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.6 python3-pip vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary dependencies for PostgreSQL and Rust
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev libpq-dev libclang-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Rust and init the cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
    /bin/bash -c "source $HOME/.cargo/env && cargo install cargo-pgrx --version '0.9.7' --locked" && \
    /bin/bash -c "source $HOME/.cargo/env && cargo pgrx init --pg14 /usr/bin/pg_config"

# Set environment variables for Rust and Python
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/project/TRAILS/internal/ml/model_selection"

WORKDIR /project
COPY ./internal/ml/model_selection/requirement.txt ./requirement.txt
RUN pip install -r requirement.txt

WORKDIR /project/TRAILS/internal/pg_extension
CMD cargo pgrx run pg14
