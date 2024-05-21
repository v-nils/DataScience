FROM python:3.11

WORKDIR /DataScience

COPY . /DataScience

RUN pip install --no-cache-dir -r requirements.txt