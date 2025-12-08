# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# install system deps (for streamlit/plotly)
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

ENV PORT=8501
EXPOSE 8501

# start script will handle scheduler + streamlit
CMD ["bash", "start.sh"]
