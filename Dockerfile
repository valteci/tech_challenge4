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

ENV FLASK_ENV=production
ENV FLASK_APP=src.app
#ENV GIT_PYTHON_REFRESH=quiet

EXPOSE 5000

CMD ["gunicorn", "src.app:app", "--bind", "0.0.0.0:5000", "-w", "1", "--timeout", "600"]
