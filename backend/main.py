from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.routes import router as core_router

app = FastAPI()
app.include_router(core_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
