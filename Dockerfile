FROM python:3.7-slim

COPY . /usr/src/app

WORKDIR /usr/src/app

RUN apt-get update && \
    pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]