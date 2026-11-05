"""FastAPI: /health, /predict, /explain."""
from fastapi import FastAPI

app = FastAPI(title="Diabetes AI Lab", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
