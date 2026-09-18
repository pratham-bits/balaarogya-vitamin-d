from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.inference import router as inference_router

app = FastAPI(
    title="BalAarogya Vitamin-D API",
    description="API for Vitamin-D risk screening prototype.",
    version="0.1.0",
)

# Allow Flutter Web development servers running on localhost/127.0.0.1
# regardless of the dynamically assigned port.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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