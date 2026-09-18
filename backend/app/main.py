"""
Image Detection FastAPI Application Entrypoint
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

from backend.app.core.config import settings
from backend.app.api.routes import health, upload, analysis, history, reports, storage, settings as backup_settings
from backend.app.services.predictor_service import predictor_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure ML model is loaded
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...")
    if not predictor_service.is_loaded():
        predictor_service.load_model()
    yield
    print(f"Shutting down {settings.APP_NAME}...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-generated image detection platform: REAL IMAGE versus AI-GENERATED IMAGE.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"^https://image-detection-frontend-wine\.vercel\.app$|^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers adhering to PRD Section 34 & 63
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": detail
            }
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": "ERROR" if exc.status_code != 401 else "UNAUTHORIZED",
                "message": str(detail)
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request payload or parameters.",
                "details": exc.errors()
            }
        }
    )

# Include API routes
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(upload.router, prefix=settings.API_PREFIX)
app.include_router(analysis.router, prefix=settings.API_PREFIX)
app.include_router(history.router, prefix=settings.API_PREFIX)
app.include_router(reports.router, prefix=settings.API_PREFIX)
app.include_router(storage.router, prefix=settings.API_PREFIX)
app.include_router(backup_settings.router, prefix=settings.API_PREFIX)

@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs"
    }
