from fastapi import FastAPI
from api.routes.inference import router as inference_router
app = FastAPI(
    title="BalAarogya Vitamin-D API",
    description="API for Vitamin-D risk screening prototype.",
    version="0.1.0",
)
app.include_router(inference_router)


@app.get("/")
def root():
    return {
        "service": "BalAarogya Vitamin-D API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }