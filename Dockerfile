FROM python:3.6

WORKDIR /sentimentAnalysis

ENV FLASK_APP=app.py

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app/app.py"]
