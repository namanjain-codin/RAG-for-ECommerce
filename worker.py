"""
worker.py
---------
Core RAG pipeline with:
  1. Hybrid Retrieval (BM25 + FAISS + RRF)
  2. Constraint-Aware Reranking
  3. Product Comparison (/compare endpoint)
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"
DOCS_CACHE_FILE  = "docs_cache.json"

RERANK_WEIGHTS = {
    "semantic":      0.40,
    "price_fit":     0.30,
    "feature_match": 0.20,
    "rating":        0.10,
}


# ── SECTION 1: Service Initialization ────────────────────────────────────────

def load_services() -> Dict[str, Any]:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is missing from .env")

    print("🔗 Loading Mistral embeddings...")
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    print("📂 Loading FAISS index...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )

    print("📖 Loading document cache for BM25...")
    with open(DOCS_CACHE_FILE, "r") as f:
        docs_cache = json.load(f)

    all_docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in docs_cache
    ]

    print("🔨 Building BM25 index...")
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"✅ BM25 index built over {len(all_docs)} documents.")

    print("🤖 Loading Mistral LLM...")
    llm = ChatMistralAI(
        model="mistral-large-latest", api_key=api_key, temperature=0.2
    )

    print("✅ All services loaded.")
    return {
        "vectorstore": vectorstore,
        "bm25_index":  bm25_index,
        "all_docs":    all_docs,
        "llm":         llm
    }


# ── SECTION 2: Constraint Extraction ─────────────────────────────────────────

CONSTRAINT_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""
You are a shopping assistant. Extract structured constraints from the user's shopping query.

User Query: "{query}"

Respond ONLY with a valid JSON object — no explanation, no markdown:
{{
  "max_price": <number or null>,
  "min_price": <number or null>,
  "min_rating": <number or null>,
  "category": <string or null>,
  "required_features": [<list of strings>],
  "use_case": "<brief description of what the user wants to do>",
  "search_query": "<clean rephrased search query for semantic search>"
}}
"""
)


def extract_constraints(query: str, llm: ChatMistralAI) -> Dict[str, Any]:
    print(f"\n🔍 Extracting constraints from query: '{query}'")
    chain  = CONSTRAINT_EXTRACTION_PROMPT | llm
    raw    = chain.invoke({"query": query}).content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    try:
        constraints = json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️ Could not parse constraints JSON. Using empty constraints.")
        constraints = {
            "max_price": None, "min_price": None, "min_rating": None,
            "category": None, "required_features": [],
            "use_case": query, "search_query": query
        }

    print(f"✅ Extracted constraints: {json.dumps(constraints, indent=2)}")
    return constraints


# ── SECTION 3: BM25 Search ────────────────────────────────────────────────────

def bm25_search(
    query: str,
    bm25_index: BM25Okapi,
    all_docs: List[Document],
    k: int = 10
) -> List[Tuple[Document, float]]:
    print(f"\n🔑 Running BM25 keyword search for: '{query}'")
    scores     = bm25_index.get_scores(query.lower().split())
    scored     = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
    print(f"✅ BM25 returned {len(scored[:k])} candidates.")
    return scored[:k]


# ── SECTION 4: FAISS Semantic Search ─────────────────────────────────────────

def semantic_search(
    search_query: str,
    vectorstore: FAISS,
    k: int = 10
) -> List[Tuple[Document, float]]:
    print(f"\n🔎 Running FAISS semantic search for: '{search_query}'")
    results = vectorstore.similarity_search_with_score(search_query, k=k)
    results = sorted(results, key=lambda x: x[1])
    print(f"✅ FAISS returned {len(results)} candidates.")
    return results


# ── SECTION 5: Reciprocal Rank Fusion ────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Document, float]],
    semantic_results: List[Tuple[Document, float]],
    k: int = 60
) -> List[Tuple[Document, float]]:
    print("\n🔀 Applying Reciprocal Rank Fusion (RRF)...")

    rrf_scores:       Dict[str, float] = {}
    faiss_similarity: Dict[str, float] = {}
    doc_map:          Dict[str, Document] = {}

    for rank, (doc, _) in enumerate(bm25_results):
        did = doc.metadata["id"]
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (rank + k)
        doc_map[did]    = doc

    max_dist = max((s for _, s in semantic_results), default=1.0) or 1.0
    for rank, (doc, dist) in enumerate(semantic_results):
        did = doc.metadata["id"]
        rrf_scores[did]       = rrf_scores.get(did, 0.0) + 1.0 / (rank + k)
        faiss_similarity[did] = 1.0 - (dist / max_dist)
        doc_map[did]          = doc

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    overlap    = {d.metadata["id"] for d, _ in bm25_results} & \
                 {d.metadata["id"] for d, _ in semantic_results}

    print(f"✅ RRF merged {len(sorted_ids)} unique candidates.")
    print(f"   📌 {len(overlap)} products appeared in BOTH retrievers (boosted).")

    return [
        (doc_map[did], faiss_similarity.get(did, 0.5))
        for did in sorted_ids
    ]


# ── SECTION 6: Constraint-Aware Reranking ────────────────────────────────────

def compute_price_fit(price: float, constraints: Dict[str, Any]) -> float:
    max_p = constraints.get("max_price")
    min_p = constraints.get("min_price")
    if max_p is None and min_p is None:
        return 1.0
    if max_p is not None and price > max_p:
        return 0.0
    if min_p is not None and price < min_p:
        return 0.0
    if max_p and max_p > 0:
        return min(price / max_p, 1.0)
    return 1.0


def compute_feature_match(
    features: List[str], tags: List[str], required: List[str]
) -> float:
    if not required:
        return 1.0
    combined = [f.lower() for f in features + tags]
    matched  = sum(1 for r in required if any(r.lower() in item for item in combined))
    return matched / len(required)


def compute_rating_score(rating: float, max_rating: float = 5.0) -> float:
    return rating / max_rating


def constraint_aware_rerank(
    candidates: List[Tuple[Document, float]],
    constraints: Dict[str, Any]
) -> List[Tuple[Document, float, Dict[str, float]]]:
    print("\n⚖️  Running Constraint-Aware Reranking...")
    required = constraints.get("required_features", [])
    scored   = []

    for doc, semantic_sim in candidates:
        meta          = doc.metadata
        price_fit     = compute_price_fit(meta["price"], constraints)
        feature_match = compute_feature_match(meta["features"], meta["tags"], required)
        rating_score  = compute_rating_score(meta["rating"])

        final_score = (
            RERANK_WEIGHTS["semantic"]      * semantic_sim  +
            RERANK_WEIGHTS["price_fit"]     * price_fit     +
            RERANK_WEIGHTS["feature_match"] * feature_match +
            RERANK_WEIGHTS["rating"]        * rating_score
        )

        breakdown = {
            "final_score":   round(final_score,    4),
            "semantic":      round(semantic_sim,   4),
            "price_fit":     round(price_fit,      4),
            "feature_match": round(feature_match,  4),
            "rating":        round(rating_score,   4),
        }

        scored.append((doc, final_score, breakdown))
        print(
            f"   📊 {meta['name'][:35]:<35} "
            f"score={final_score:.3f} "
            f"[sem={semantic_sim:.2f} price={price_fit:.2f} "
            f"feat={feature_match:.2f} rating={rating_score:.2f}]"
        )

    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"✅ Reranking complete. Top: {scored[0][0].metadata['name']}")
    return scored


# ── SECTION 7: Hybrid Search ──────────────────────────────────────────────────

def hybrid_search(
    query: str,
    search_query: str,
    services: Dict[str, Any],
    k: int = 10
) -> List[Tuple[Document, float]]:
    bm25_r     = bm25_search(query, services["bm25_index"], services["all_docs"], k=k)
    semantic_r = semantic_search(search_query, services["vectorstore"], k=k)
    return reciprocal_rank_fusion(bm25_r, semantic_r)


# ── SECTION 8: Explanation Generation ────────────────────────────────────────

EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["user_query", "use_case", "product_details",
                     "constraints_summary", "score_breakdown"],
    template="""
You are an expert shopping assistant.

User Query: "{user_query}"
Use Case: "{use_case}"
Active Filters: {constraints_summary}

Product:
{product_details}

Relevance Score Breakdown (out of 1.0):
{score_breakdown}

Write a SHORT, helpful explanation (2-3 sentences) of why this product is a good match.
- Mention specific features that match their needs
- Reference the score breakdown naturally if relevant
- Be conversational and helpful, not salesy
- Do NOT start with the product name
"""
)


def generate_explanation(
    doc: Document,
    breakdown: Dict[str, float],
    query: str,
    constraints: Dict[str, Any],
    llm: ChatMistralAI
) -> str:
    meta = doc.metadata
    cs   = []
    if constraints.get("max_price"):
        cs.append(f"budget under ₹{constraints['max_price']}")
    if constraints.get("required_features"):
        cs.append(f"must have: {', '.join(constraints['required_features'])}")
    if constraints.get("category"):
        cs.append(f"category: {constraints['category']}")

    product_details = (
        f"Name: {meta['name']}\nBrand: {meta['brand']}\n"
        f"Price: ₹{meta['price']}\nRating: {meta['rating']}/5\n"
        f"Features: {', '.join(meta['features'])}\n"
        f"Description: {doc.page_content.split('Description:')[-1].strip()}"
    )
    score_text = "\n".join(
        f"  - {k.replace('_',' ').title()}: {v}"
        for k, v in breakdown.items() if k != "final_score"
    )

    chain    = EXPLANATION_PROMPT | llm
    response = chain.invoke({
        "user_query":          query,
        "use_case":            constraints.get("use_case", query),
        "product_details":     product_details,
        "constraints_summary": ", ".join(cs) or "none",
        "score_breakdown":     score_text,
    })
    return response.content.strip()


# ── SECTION 9: Product Comparison ────────────────────────────────────────────

def _find_product_by_name(
    name: str, all_docs: List[Document]
) -> Optional[Document]:
    """
    Finds a product document by fuzzy name match.
    Tries exact match first, then partial match.
    """
    name_lower = name.lower().strip()

    # Exact match
    for doc in all_docs:
        if doc.metadata["name"].lower() == name_lower:
            return doc

    # Partial match — return first document whose name contains the query
    for doc in all_docs:
        if name_lower in doc.metadata["name"].lower():
            return doc

    # Reverse partial — product name contained in query string
    for doc in all_docs:
        if doc.metadata["name"].lower() in name_lower:
            return doc

    return None


COMPARISON_PROMPT = PromptTemplate(
    input_variables=["use_case", "products_block"],
    template="""
You are an expert product analyst and shopping advisor.

A user wants to compare the following products for this use case:
Use Case / Intent: "{use_case}"

Here are the products:
{products_block}

Produce a structured comparison in the following EXACT JSON format.
Respond ONLY with valid JSON — no markdown, no explanation outside the JSON:

{{
  "summary": "<2-3 sentence overall summary of how these products compare for the use case>",
  "comparison_table": [
    {{
      "product_id": "<id>",
      "product_name": "<name>",
      "price": <number>,
      "rating": <number>,
      "pros": ["<pro 1>", "<pro 2>", "<pro 3>"],
      "cons": ["<con 1>", "<con 2>"],
      "best_for": "<one sentence — who or what situation this product is best for>",
      "use_case_fit": "<high | medium | low> — how well it fits the stated use case",
      "verdict": "<one sentence recommendation>"
    }}
  ],
  "winner": {{
    "product_id": "<id of the best overall pick>",
    "reason": "<why this product wins for the stated use case>"
  }}
}}
"""
)


def compare_products(
    product_names: List[str],
    use_case: str,
    services: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Finds each named product in the catalog, then uses Mistral LLM
    to produce a structured comparison table with pros, cons,
    use-case fit, and an overall winner for the stated use case.
    """
    llm      = services["llm"]
    all_docs = services["all_docs"]

    # Step 1: Resolve product names to documents
    print(f"\n🔍 Resolving {len(product_names)} product names...")
    resolved: List[Document] = []
    not_found: List[str]     = []

    for name in product_names:
        doc = _find_product_by_name(name, all_docs)
        if doc:
            resolved.append(doc)
            print(f"  ✅ Found: {doc.metadata['name']}")
        else:
            not_found.append(name)
            print(f"  ❌ Not found: '{name}'")

    if len(resolved) < 2:
        raise ValueError(
            f"Need at least 2 valid products to compare. "
            f"Could not find: {not_found}"
        )

    # Step 2: Build products block for the prompt
    products_block = ""
    for doc in resolved:
        meta = doc.metadata
        products_block += (
            f"\n---\n"
            f"ID: {meta['id']}\n"
            f"Name: {meta['name']}\n"
            f"Brand: {meta['brand']}\n"
            f"Category: {meta['category']}\n"
            f"Price: ₹{meta['price']}\n"
            f"Rating: {meta['rating']}/5\n"
            f"Features: {', '.join(meta['features'])}\n"
            f"Description: {doc.page_content.split('Description:')[-1].strip()}\n"
        )

    # Step 3: Call LLM for structured comparison
    print(f"\n🤖 Generating comparison for {len(resolved)} products...")
    chain    = COMPARISON_PROMPT | llm
    response = chain.invoke({
        "use_case":      use_case,
        "products_block": products_block
    })
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    try:
        comparison = json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️ Could not parse comparison JSON from LLM.")
        comparison = {"raw_response": raw}

    print("✅ Comparison generated.")

    return {
        "use_case":        use_case,
        "products_compared": [doc.metadata["name"] for doc in resolved],
        "not_found":       not_found,
        "comparison":      comparison
    }


# ── SECTION 10: Main Recommendation Pipeline ──────────────────────────────────

def get_recommendations(
    query: str,
    services: Dict[str, Any],
    top_n: int = 5
) -> Dict[str, Any]:
    llm = services["llm"]

    constraints  = extract_constraints(query, llm)
    search_query = constraints.get("search_query", query)

    merged    = hybrid_search(query, search_query, services, k=10)
    reranked  = constraint_aware_rerank(merged, constraints)
    top_results = reranked[:top_n]

    print(f"\n✍️ Generating explanations for {len(top_results)} products...")
    recommendations = []
    for doc, final_score, breakdown in top_results:
        meta        = doc.metadata
        explanation = generate_explanation(doc, breakdown, query, constraints, llm)
        recommendations.append({
            "id":              meta["id"],
            "name":            meta["name"],
            "brand":           meta["brand"],
            "category":        meta["category"],
            "price":           meta["price"],
            "currency":        meta["currency"],
            "rating":          meta["rating"],
            "features":        meta["features"],
            "score_breakdown": breakdown,
            "explanation":     explanation
        })
        print(f"  ✅ Explained: {meta['name']} (score={final_score:.3f})")
        time.sleep(2)

    return {
        "query":               query,
        "extracted_constraints": constraints,
        "retrieval_info": {
            "method":                        "Hybrid (BM25 + FAISS + RRF) + Constraint-Aware Reranking",
            "rerank_weights":                RERANK_WEIGHTS,
            "total_candidates_before_rerank": len(merged),
        },
        "recommendations": recommendations
    }
