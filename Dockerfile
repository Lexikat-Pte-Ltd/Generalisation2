FROM python:3.10
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["tail","-f"]