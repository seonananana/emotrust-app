\from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze, posts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/analyze")
app.include_router(posts.router, prefix="/posts")
