"""
api.py
------
FastAPI server with:
  POST /recommend  - Hybrid RAG recommendations with reranking
  POST /compare    - Structured product comparison table
  GET  /           - Health check
"""

from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from worker import load_services, get_recommendations, compare_products

services: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting E-commerce Hybrid Recommendation API...")
    services.update(load_services())
    print("✅ API is ready!")
    yield
    print("👋 Shutting down...")
    services.clear()


app = FastAPI(
    title="🛒 E-commerce Hybrid RAG Recommendation Engine",
    description=(
        "Natural language recommendations (BM25 + FAISS + RRF + Reranking) "
        "and structured product comparison with pros/cons."
    ),
    version="4.0.0",
    lifespan=lifespan
)


# ── Pydantic Models — /recommend ──────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = Field(..., description="Natural language shopping query")
    top_n: int = Field(default=3, ge=1, le=10)


class ScoreBreakdown(BaseModel):
    final_score:   float
    semantic:      float
    price_fit:     float
    feature_match: float
    rating:        float


class ProductRecommendation(BaseModel):
    id:              str
    name:            str
    brand:           str
    category:        str
    price:           float
    currency:        str
    rating:          float
    features:        List[str]
    score_breakdown: ScoreBreakdown
    explanation:     str


class ExtractedConstraints(BaseModel):
    max_price:         Optional[float] = None
    min_price:         Optional[float] = None
    min_rating:        Optional[float] = None
    category:          Optional[str]   = None
    required_features: List[str]       = []
    use_case:          Optional[str]   = None
    search_query:      Optional[str]   = None


class RetrievalInfo(BaseModel):
    method:                         str
    rerank_weights:                 Dict[str, float]
    total_candidates_before_rerank: int


class RecommendResponse(BaseModel):
    query:                 str
    extracted_constraints: ExtractedConstraints
    retrieval_info:        RetrievalInfo
    recommendations:       List[ProductRecommendation]


# ── Pydantic Models — /compare ────────────────────────────────────────────────

class CompareRequest(BaseModel):
    product_names: List[str] = Field(
        ...,
        min_length=2,
        description="List of 2-4 product names to compare (exact or partial match)",
        examples=[["AquaShield Hiking Boots", "RainGuard Jacket Pro", "NightHike Headlamp 300LM"]]
    )
    use_case: str = Field(
        ...,
        description="What the user intends to use the products for",
        examples=["3-day hiking trip in monsoon season"]
    )


class ComparisonTableEntry(BaseModel):
    product_id:   str
    product_name: str
    price:        float
    rating:       float
    pros:         List[str]
    cons:         List[str]
    best_for:     str
    use_case_fit: str   # high | medium | low
    verdict:      str


class Winner(BaseModel):
    product_id: str
    reason:     str


class ComparisonResult(BaseModel):
    summary:          str
    comparison_table: List[ComparisonTableEntry]
    winner:           Winner


class CompareResponse(BaseModel):
    use_case:          str
    products_compared: List[str]
    not_found:         List[str] = Field(
        default=[],
        description="Product names that could not be matched in the catalog"
    )
    comparison:        ComparisonResult


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message":  "🛒 E-commerce Hybrid RAG Recommendation Engine is live!",
        "endpoints": {
            "POST /recommend": "Natural language product recommendations",
            "POST /compare":   "Structured comparison table for 2-4 products"
        }
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Get product recommendations from a natural language query.
    Uses BM25 + FAISS + RRF + Constraint-Aware Reranking.
    Each result includes a score_breakdown for full transparency.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        return get_recommendations(
            query=request.query,
            services=services,
            top_n=request.top_n
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    """
    Compare 2-4 products side by side for a specific use case.

    The LLM produces:
    - A summary of how the products compare
    - A structured table with pros, cons, use-case fit, and verdict per product
    - An overall winner with reasoning

    Example request:
    {
      "product_names": ["AquaShield Hiking Boots", "RainGuard Jacket Pro"],
      "use_case": "3-day hiking trip in the monsoon season"
    }
    """
    if len(request.product_names) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 product names.")
    if len(request.product_names) > 4:
        raise HTTPException(status_code=400, detail="Compare at most 4 products at a time.")
    if not request.use_case.strip():
        raise HTTPException(status_code=400, detail="use_case cannot be empty.")

    try:
        return compare_products(
            product_names=request.product_names,
            use_case=request.use_case,
            services=services
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
