from pathlib import Path

from fastapi import FastAPI

app = FastAPI()

TRANSCRIPT_PATH = Path(__file__).resolve().parent.parent / "data" / "transcript.txt"


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/search")
async def search(q: str):
    text = TRANSCRIPT_PATH.read_text()
    lines = [line for line in text.splitlines() if q.lower() in line.lower()]
    return {"query": q, "results": lines}