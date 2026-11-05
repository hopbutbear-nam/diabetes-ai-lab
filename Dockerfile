FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .
EXPOSE 8200
CMD ["uvicorn", "diabetes_ai.serving.app:app", "--host", "0.0.0.0", "--port", "8200"]
