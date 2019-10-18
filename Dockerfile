FROM python:3.6-slim
COPY . /deploy/
WORKDIR /deploy/ 
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]