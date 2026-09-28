"""
Main FastAPI Application Entrypoint
Project: Demand-Decision-Intelligence
Location: backend/main.py
"""

import time
import uuid
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

from backend.core.config import settings
from backend.core.logging_config import (
    setup_logging,
    request_id_ctx,
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION,
)
from backend.api import (
    health,
    auth,
    demand,
    products,
    upload,
    forecast,
    inventory,
    analytics,
    market_prices,
    pricing,
    admin,
    procurement,
    recommendations,
    alerts,
    transfers,
    classification,
    dead_stock,
    calendar,
    explain,
    assistant,
    digests,
    quality,
    chat,
)

# Initialize structured logging globally
setup_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
)


@app.on_event("startup")
def startup_checks():
    from backend.db.session import SessionLocal
    from backend.services.dataset_service import check_demo_dataset_integrity
    db = SessionLocal()
    try:
        res = check_demo_dataset_integrity(db)
        if res.get("warning"):
            logger.warning("startup_demo_dataset_warning", warning=res["warning"])
    except Exception as e:
        logger.error("startup_checks_error", error=str(e))
    finally:
        db.close()


# Request tracing & metrics middleware
@app.middleware("http")
async def request_tracing_and_metrics_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_ctx.set(req_id)
    request.state.request_id = req_id

    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # Instrument Prometheus metrics
        endpoint = request.url.path
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time"] = f"{(duration * 1000.0):.2f}ms"
        return response
    except Exception as exc:
        duration = time.time() - start_time
        logger.exception("unhandled_request_exception", error=str(exc), request_id=req_id)
        origin = request.headers.get("origin")
        error_headers = {"X-Request-ID": req_id}
        if origin:
            error_headers["Access-Control-Allow-Origin"] = origin
            error_headers["Access-Control-Allow-Credentials"] = "true"
            error_headers["Access-Control-Allow-Methods"] = "*"
            error_headers["Access-Control-Allow-Headers"] = "*"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_id": req_id,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": f"Internal error: {exc}",
            },
            headers=error_headers,
        )
    finally:
        request_id_ctx.reset(token)


# Hardened CORS configuration (Outer layer to always apply CORS headers even on errors)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API Routers
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Health"])
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Auth"])
app.include_router(demand.router, prefix=f"{settings.API_V1_STR}/demand", tags=["Demand Intelligence"])
app.include_router(products.router, prefix=f"{settings.API_V1_STR}/products", tags=["Products"])
app.include_router(upload.router, prefix=f"{settings.API_V1_STR}/upload", tags=["Data Upload"])
app.include_router(forecast.router, prefix=f"{settings.API_V1_STR}/forecast", tags=["Forecasting Suite"])
app.include_router(inventory.router, prefix=f"{settings.API_V1_STR}/inventory", tags=["Inventory Optimization"])
app.include_router(analytics.router, prefix=f"{settings.API_V1_STR}/analytics", tags=["Analytics & Anomalies"])
app.include_router(pricing.router, prefix=f"{settings.API_V1_STR}/pricing", tags=["Price Elasticity & Optimization"])
app.include_router(market_prices.router, prefix=f"{settings.API_V1_STR}/market-prices", tags=["Government Market Prices"])
app.include_router(procurement.router, prefix=settings.API_V1_STR, tags=["Procurement & Purchase Orders"])
app.include_router(recommendations.router, prefix=settings.API_V1_STR, tags=["Recommendations & Learning Loop"])
app.include_router(alerts.router, prefix=settings.API_V1_STR, tags=["Alerting Engine"])
app.include_router(transfers.router, prefix=settings.API_V1_STR, tags=["Inter-City Transfers"])
app.include_router(classification.router, prefix=f"{settings.API_V1_STR}/classification", tags=["ABC-XYZ Classification"])
app.include_router(dead_stock.router, prefix=f"{settings.API_V1_STR}/dead-stock", tags=["Dead Stock & Markdown Actions"])
app.include_router(calendar.router, prefix=f"{settings.API_V1_STR}/calendar", tags=["Indian Festival Calendar & Regressors"])
app.include_router(explain.router, prefix=f"{settings.API_V1_STR}/explain", tags=["Per-SKU Explainability"])
app.include_router(assistant.router, prefix=f"{settings.API_V1_STR}/assistant", tags=["Guarded NL Assistant"])
app.include_router(digests.router, prefix=f"{settings.API_V1_STR}/digests", tags=["Weekly Narrative Digests"])
app.include_router(quality.router, prefix=f"{settings.API_V1_STR}/quality", tags=["Data Quality Scorecard"])
app.include_router(admin.router, prefix=settings.API_V1_STR, tags=["Observability & Admin"])
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["AI Decision Assistant"])


@app.get("/")
def root():
    return {
        "message": "Welcome to Demand & Decision Intelligence System API",
        "docs": f"{settings.API_V1_STR}/docs",
        "health": f"{settings.API_V1_STR}/health",
        "deep_health": f"{settings.API_V1_STR}/health/deep",
        "metrics": f"{settings.API_V1_STR}/metrics",
    }
