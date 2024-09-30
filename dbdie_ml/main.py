"""Main FastAPI API."""

from fastapi import FastAPI

from backbone.routers import cropping, extraction, training

app = FastAPI(
    title="DBDIE ML API",
    summary="DBD Information Extraction ML API",
    description="ML package to process your 💀 Dead By Daylight 💀 matches' endcards.",
)

app.include_router(cropping.router, prefix="/crop")
app.include_router(extraction.router, prefix="/extract")
app.include_router(training.router, prefix="/train")


@app.get("/health", summary="Health check")
def health():
    return {"status": "OK"}
