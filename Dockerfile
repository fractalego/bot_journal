ARG PYTHON_VERSION="3.6"
FROM python:${PYTHON_VERSION} AS builder

RUN apt-get update
RUN apt-get install -qq -y curl
RUN apt-get install -qq -y build-essential python-dev
RUN apt-get install -qq -y wget

WORKDIR /usr/local/
COPY src src/
COPY data data/
COPY models models/
COPY cache cache/
COPY templates templates/

COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN python -m nltk.downloader punkt

ENV PORT="8010"
EXPOSE ${PORT}

CMD ["python", "-m", "src.server"]