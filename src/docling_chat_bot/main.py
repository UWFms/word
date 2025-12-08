from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .app.api.routes import router

app = FastAPI(
    title="Docling API",
    version="1.0.0"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)
