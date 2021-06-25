FROM python:3.8.5
COPY . .
RUN pip3 install -r requirements.txt
