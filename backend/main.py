# backend/main.py
"""
FastAPI приложение с правильной инициализацией SAM2.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
from core.config import device as DEVICE
from routers import (
    init_router,
    segment_router,
    results_router,
    control_router,
    analysis_router
)
from utils.debug import print_debug
from services.sam2_service import initialize_sam2


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Правильная инициализация SAM2 моделей при старте приложения.
    """
    # === ЗАГРУЗКА SAM2 ВНУТРИ LIFESPAN ===
    print_debug("Initializing SAM2 models...")
        
    # Явно вызываем инициализацию
    sam2_objects = initialize_sam2()
    
    # Сохраняем в app.state для доступа из всех роутеров
    app.state.sam2_objects = sam2_objects
    
    print_debug(f"SAM2 models initialized successfully on device: {DEVICE}")
    print_debug(f"Available objects: {list(sam2_objects.keys())}")

    yield

    # === ОЧИСТКА ПРИ ВЫКЛЮЧЕНИИ ===
    print_debug("Cleaning up SAM2 models...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_debug("Cleanup completed")


# === СОЗДАНИЕ APP С LIFESPAN ===
app = FastAPI(
    title="Snowflake Analysis API",
    description="API for snowflake segmentation and analysis using SAM2",
    version="1.0.0",
    lifespan=lifespan
)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ПОДКЛЮЧЕНИЕ РОУТЕРОВ ===
app.include_router(init_router, prefix="/api", tags=["initialization"])
app.include_router(segment_router, prefix="/api", tags=["segmentation"]) 
app.include_router(results_router, prefix="/api", tags=["results"])
app.include_router(control_router, prefix="/api", tags=["control"])
app.include_router(analysis_router, prefix="/api", tags=["analysis"])


# === HEALTH CHECK ===
@app.get("/health")
async def health():
    """Проверка статуса приложения и моделей"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": "sam2" if hasattr(app, 'state') and hasattr(app.state, 'sam2_objects') else "none"
    }

'''
@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о API"""
    return {
        "message": "Snowflake Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# === ЗАПУСК ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Только для разработки
        log_level="debug"
    )
'''   