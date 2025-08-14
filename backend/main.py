# backend/main.py
from fastapi import FastAPI
from routers.analyze import router as analyze_router
from routers.posts import router as posts_router

app = FastAPI()

# 라우터 등록
app.include_router(analyze_router, prefix="/analyze", tags=["Analyze"])
app.include_router(posts_router, prefix="/posts", tags=["Posts"])

@app.get("/")
def root():
    return {"msg": "Emotrust API is running"}
