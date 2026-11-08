FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY app ./app
COPY knowledge_base ./knowledge_base
RUN pip install --no-cache-dir -e . --no-deps
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8200
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8200/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "diabetes_ai.serving.app:app", "--host", "0.0.0.0", "--port", "8200"]
