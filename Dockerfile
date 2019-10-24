FROM python:3.6-slim
WORKDIR /deploy/ 

COPY requirements.txt . 

RUN apk add --no-cache --virtual .build-deps \
    build-base openssl-dev pkgconfig libffi-dev \
    cups-dev jpeg-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apk del .build-deps

COPY . . 
EXPOSE 403
ENTRYPOINT ["python", "app.py"]
