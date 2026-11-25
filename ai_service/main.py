from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from .services.model_loader import model_loader
    from .routers.predictions import router as predictions_router
except ImportError:
    from services.model_loader import model_loader
    from routers.predictions import router as predictions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 기동 시 모델을 메모리에 로드
    print("[FastAPI] 서버를 시작합니다. AI 모델을 메모리에 로드합니다...")
    model_loader.load_model()
    print("[FastAPI] AI 모델 로드 완료. 추론 준비 완료!")
    yield
    print("[FastAPI] 서버가 안전하게 종료되었습니다.")


app = FastAPI(
    title="Vision Action AI Service",
    description="YOLO Pose 및 Random Forest 기반 동작 추론 서비스 (교안 6개 동작 기준)",
    version="1.0.0",
    lifespan=lifespan,
)

# 브라우저 및 Django 연동을 위한 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 경로 접속 시 Swagger API 문서로 자동 리다이렉트
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# 헬스 체크 엔드포인트
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "ai_service",
        "model_loaded": model_loader.is_loaded,
        "classes": model_loader.classes,
    }


# 추론 라우터 연결 (루트 경로 /predict/video 및 /api/v1/predict 둘 다 지원)
app.include_router(predictions_router)
app.include_router(predictions_router, prefix="/api/v1")
