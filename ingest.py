"""
ingest.py
---------
Run this ONCE to index your product catalog into a local FAISS vector store.
Also saves raw product documents as JSON so BM25 can use them at runtime.

Usage: python ingest.py
"""

import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PRODUCTS_FILE = "products.json"
FAISS_INDEX_PATH = "faiss_index"
DOCS_CACHE_FILE = "docs_cache.json"  # NEW: saved for BM25 at runtime


def build_product_document(product: dict) -> Document:
    """
    Converts a product dict into a LangChain Document.
    The page_content is a rich text description used for semantic search.
    All raw fields are stored in metadata for constraint filtering.
    """
    content = (
        f"Product: {product['name']}\n"
        f"Brand: {product['brand']}\n"
        f"Category: {product['category']}\n"
        f"Price: ₹{product['price']}\n"
        f"Rating: {product['rating']} / 5\n"
        f"Features: {', '.join(product['features'])}\n"
        f"Tags: {', '.join(product['tags'])}\n"
        f"Description: {product['description']}"
    )

    metadata = {
        "id": product["id"],
        "name": product["name"],
        "brand": product["brand"],
        "category": product["category"],
        "price": product["price"],
        "currency": product["currency"],
        "rating": product["rating"],
        "features": product["features"],
        "tags": product["tags"],
    }

    return Document(page_content=content, metadata=metadata)


def ingest():
    print("📦 Loading product catalog...")
    with open(PRODUCTS_FILE, "r") as f:
        products = json.load(f)
    print(f"✅ Loaded {len(products)} products.")

    print("📝 Building LangChain documents...")
    docs = [build_product_document(p) for p in products]

    # --- NEW: Save docs to JSON cache for BM25 ---
    print(f"💾 Saving document cache to '{DOCS_CACHE_FILE}' for BM25...")
    docs_cache = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(DOCS_CACHE_FILE, "w") as f:
        json.dump(docs_cache, f, indent=2)
    print(f"✅ Saved {len(docs_cache)} documents to cache.")

    print("🔗 Connecting to Mistral Embeddings API...")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    print("🧠 Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print(f"💾 Saving FAISS index to '{FAISS_INDEX_PATH}'...")
    vectorstore.save_local(FAISS_INDEX_PATH)

    print("✅ Ingestion complete! You can now start the API server.")


if __name__ == "__main__":
    ingest()
