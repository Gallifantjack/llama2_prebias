# Use an official Python runtime as the parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /workspace

# Accept arguments during build
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL

# Set git config values
RUN apt-get update && apt-get install -y git
RUN git config --global user.name "${GIT_USER_NAME}" && git config --global user.email "${GIT_USER_EMAIL}"

# Install any needed packages specified in requirements.txt
COPY requirements.txt /workspace/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Keep the container running
CMD ["tail", "-f", "/dev/null"]
