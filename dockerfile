FROM python:3.10

ARG PYTHON_VERSION=3.10
ARG PYG_URL=https://data.pyg.org/whl/torch-2.0.0+cu118.html
# RUN apt update \
#     && apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils python3-pip \
#     && cd /usr/local/bin \
#     && ln -s /usr/bin/python${PYTHON_VERSION} python

RUN  python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN python -m pip install  -r requirements.txt 

COPY requirements-torch.txt requirements-torch.txt
RUN python -m pip install -r requirements-torch.txt 

COPY requirements-pyg.txt requirements-pyg.txt
RUN python -m pip install -r requirements-pyg.txt -f ${PYG_URL}

RUN rm requirements.txt
RUN rm requirements-torch.txt
RUN rm requirements-pyg.txt




