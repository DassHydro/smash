# Derived from Pandas
FROM python:3.11
WORKDIR /home/smash

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y \
    build-essential \
    openjdk-17-jdk \
    gfortran

RUN python -m pip install --upgrade pip
COPY requirements-dev.txt /tmp
RUN python -m pip install -r /tmp/requirements-dev.txt
RUN git config --global --add safe.directory /home/smash
CMD ["/bin/bash"]