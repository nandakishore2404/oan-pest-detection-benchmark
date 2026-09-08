# -*- coding: utf-8 -*-
"""
FastAPI Application Entrypoint: OAN Kenya Pest & Disease Service
===============================================================
Runs via: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="OpenAgriNet (OAN) Kenya: Pest & Crop Disease AI Service",
    description="Beckn-ready computer-vision diagnostic microservice for African smallholder farming.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
