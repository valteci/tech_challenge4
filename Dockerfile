FROM python:3.12.3-slim

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o restante da aplicação
COPY . .

ENV FLASK_ENV=development
ENV FLASK_APP=app.py

EXPOSE 5000

# CMD [ \
#   "/bin/sh", \
#   "-c", \
#   "if [ \"$FLASK_ENV\" = 'development' ]; then \
#      flask run --host=0.0.0.0 --port=5000 --debug; \
#    else \
#      gunicorn ${FLASK_APP%.*}:app --bind 0.0.0.0:5000; \
#    fi" \
# ]

CMD ["sleep", "1d"]