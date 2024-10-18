"""Main FastAPI API."""

from fastapi import FastAPI

from backbone.options import ENDPOINTS as EP
from backbone.routers import cropping, extraction, training

app = FastAPI(
    title="DBDIE ML API",
    summary="DBD Information Extraction ML API",
    description="ML package to process your 💀 Dead By Daylight 💀 matches' endcards.",
)

app.include_router(cropping.router,   prefix=EP.CROP)
app.include_router(extraction.router, prefix=EP.EXTRACT)
app.include_router(training.router,   prefix=EP.TRAIN)


@app.get("/health", summary="Health check")
def health():
    return {"status": "OK"}


with open("dbdie_ml/ascii_art.txt") as f:
    print(f.read())
