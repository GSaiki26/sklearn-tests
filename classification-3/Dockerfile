# Basics
FROM python:3.11-slim-buster
WORKDIR /app

# Update the container
RUN apt-get update && apt-get upgrade -y;
RUN useradd --shell /bin/bash -m user; chown user /app;
USER user

# Install the packages
COPY --chown=user requirements.txt .
RUN pip3 install --no-warn-script-location -r requirements.txt

# Run the project
COPY --chown=user src ./src
CMD python3 -u src/main.py
