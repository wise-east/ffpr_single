FROM python:3.6-slim
WORKDIR /deploy/ 

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY . . 
EXPOSE 403
ENTRYPOINT ["gunicorn", "app:app", "--bind", "0.0.0.0:403", "-t", "1000", "-w", "4"]
