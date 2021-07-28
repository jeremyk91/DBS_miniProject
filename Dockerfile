# Dockerfile, Tmage, Container
FROM python:3.8


WORKDIR /dbs_minip

COPY . /dbs_minip

RUN pip install -r requirements.txt

# define environment variables
ENV PYTHONPATH=/dbs_minip


CMD ["python", "./main/flask_web_service.py"]