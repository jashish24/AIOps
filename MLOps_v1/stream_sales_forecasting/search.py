from fastapi import FastAPI, Query
from typing import Any, Dict, List
import random

app = FastAPI()

# Example in-memory "knowledge base" you can replace with real retrieval logic
KB = [
    {"text": "How to set up n8n webhooks", "source": "docs.n8n.io"},
    {"text": "Troubleshooting Docker networking for n8n", "source": "blog.example.com"},
    {"text": "FastAPI quickstart and deployment tips", "source": "fastapi.tutorial"},
    {"text": "Designing RAG systems with embeddings", "source": "ml.example.com"},
    {"text": "Best practices for OpenAI prompts", "source": "ai.example.com"},
    {"text": "Using Docker Compose for multi-service apps", "source": "devops.example.com"},
    {"text": "How to test HTTP endpoints with curl", "source": "tools.example.com"},
    {"text": "Securing webhooks with basic auth and tokens", "source": "security.example.com"},
    {"text": "Scaling retriever services with Redis cache", "source": "infra.example.com"},
    {"text": "Parsing query parameters in FastAPI", "source": "fastapi.tutorial"},
    {"text": "Example autosuggest UX patterns", "source": "ux.example.com"},
    {"text": "Rate limiting strategies for APIs", "source": "ops.example.com"},
    {"text": "Logging and observability for microservices", "source": "observability.example.com"},
    {"text": "How to run services on host vs container", "source": "docker.example.com"},
    {"text": "Sample dataset formats for retrieval", "source": "data.example.com"}
]

@app.get("/search")
def search(q: str = Query(...), limit: int = Query(10, ge=1, le=50)) -> Dict[str, Any]:
    """
    Simple retrieval stub:
    - q: query string
    - limit: max number of results returned (default 10, max 50)
    """
    # Very simple relevance simulation: include items that contain any token from q (case-insensitive)
    tokens = [t.lower() for t in q.split() if t.strip()]
    matched = []
    for item in KB:
        text_lower = item["text"].lower()
        score = 0.0
        for t in tokens:
            if t in text_lower:
                score += 1.0
        # small random boost so results vary a bit for demo
        score += random.random() * 0.2
        if score > 0 or not tokens:
            matched.append({**item, "score": round(score, 3)})

    # sort by score descending, then by source
    matched.sort(key=lambda x: (-x["score"], x["source"]))
    # limit results
    results = matched[:limit]

    # If nothing matched, return top N generic suggestions
    if not results:
        results = [
            {"text": f"General context about {q}", "source": "fallback", "score": 0.1}
            for _ in range(min(limit, 5))
        ]

    context = "\n".join(r["text"] for r in results)
    return {"query": q, "limit": limit, "results": results, "context": context}
