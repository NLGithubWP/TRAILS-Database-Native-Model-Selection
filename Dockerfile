# Use PostgreSQL 15
FROM postgres:15

# Update the system and install necessary components
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    postgresql-plpython3-15

# Set up a virtual environment and activate it
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install any Python libraries you need for your UDFs
RUN pip3 install --no-cache-dir numpy pandas

# Add setup script
COPY ./setup.sh /docker-entrypoint-initdb.d/

# CMD statement to run PostgreSQL when the container starts
CMD ["postgres"]

# docker build -t trails .
# docker run -d --name trails -v $(pwd)/TRAILS:/TRAILS -v $(pwd)/postgresdata/data:/var/lib/postgresql/data -e POSTGRES_USER=trails -e POSTGRES_PASSWORD=trails trails


