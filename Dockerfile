FROM python:3.10-slim

WORKDIR /app

# ✅ 先复制 requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY generator.py .
COPY app.py .
COPY data/ ./data/

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 5000

CMD ["python", "app.py"]