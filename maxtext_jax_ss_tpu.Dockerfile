ARG JAX_SS_BASEIMAGE

# JAX Stable Stack Base Image
From $JAX_SS_BASEIMAGE

ARG COMMIT_HASH

ENV COMMIT_HASH=$COMMIT_HASH

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .
RUN ls .

ARG USE_MAXTEXT_REQUIREMENTS_FILE

# Install MaxText requirements
RUN if [ "${USE_MAXTEXT_REQUIREMENTS_FILE}" = "true" ]; then \
        echo "Using MaxText requirements: /deps/requirements.txt" && \
        pip install -r /deps/requirements.txt; \
    else \
        echo "Not using MaxText requirements: /deps/requirements.txt"; \
    fi

# Run the script available in JAX-SS base image to generate the manifest file
RUN bash /generate_manifest.sh PREFIX=maxtext COMMIT_HASH=$COMMIT_HASH