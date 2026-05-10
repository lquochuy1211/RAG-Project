# app/db/qdrant_client.py

import hashlib
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import MatchText
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter,
    FieldCondition, MatchValue
)

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Global client instance
client = None


def get_client() -> QdrantClient:
    """Get or create Qdrant client instance."""
    global client
    if client is None:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=30
        )
        logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")
    return client


def init_collection():
    """Initialize collection if not exists."""
    try:
        qdrant = get_client()
        collections = qdrant.get_collections().collections
        collection_names = [col.name for col in collections]

        if settings.COLLECTION_NAME not in collection_names:
            qdrant.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"[OK] Created collection: {settings.COLLECTION_NAME}")
        else:
            logger.info(f"[OK] Collection already exists: {settings.COLLECTION_NAME}")

        # --- Create Indexes ---

        # 1. Index cho việc lọc nhóm (quan trọng cho RAG)
        try:
            qdrant.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name="group_id",
                field_schema="keyword",
            )
        except:
            pass

        try:
            qdrant.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name="tenancy",
                field_schema="keyword",
            )
        except:
            pass

        try:
            qdrant.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name="is_shared",
                field_schema="keyword",
            )
        except:
            pass

        try:
            qdrant.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name="text",  # Trường chứa nội dung chính
                field_schema="text",  # Schema "text" hỗ trợ token hóa
            )
            logger.info("[INDEX] Verified/Created text index for field 'text'")
        except Exception as e:
            logger.warning(f"Note on text index: {e}")

    except Exception as e:
        logger.error(f"Error initializing collection: {e}")
        raise


def search_similar(
        query_vector: List[float],
        limit: int = 6,
) -> List[dict]:
    """
    Dense search (Vector only).
    Chỉ tìm kiếm trong PUBLIC knowledge base.
    """
    try:
        qdrant = get_client()

        # Filter chỉ lấy dữ liệu PUBLIC
        shared_filter = Filter(
            must=[
                FieldCondition(key="group_id", match=MatchValue(value="PUBLIC")),
            ]
        )

        # Search vector (Qdrant tự trả về top score, không cần deduplicate thủ công)
        results = qdrant.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=shared_filter,
            limit=limit,
            with_payload=True,
        )

        logger.info(f"[DENSE] Found {len(results)} docs")

        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    except Exception as e:
        logger.error(f"[SEARCH_SIMILAR] Error: {e}")
        return []


def search_similar_hybrid(
        query_vector: List[float],
        query_text: str,
        limit: int = 5,
) -> List[dict]:
    """
    Hybrid retrieval: Vector + Keyword.
    Đã sửa lỗi logic scroll: Sử dụng MatchText để tìm đúng keyword.
    """
    try:
        logger.info(f"[HYBRID] Query: {query_text[:30]}...")
        qdrant = get_client()

        # 1. DENSE SEARCH
        dense_results = search_similar(query_vector, limit * 2)

        # 2. SPARSE SEARCH (Keyword Match)
        sparse_results = []
        # Chỉ lấy các từ khóa dài > 2 ký tự
        keywords = [kw for kw in query_text.lower().split() if len(kw) > 2][:5]

        for keyword in keywords:
            try:
                # FILTER: Tìm keyword xuất hiện trong trường "text" hoặc "answer"
                # Yêu cầu: Field "text" hoặc "answer" nên được đánh index dạng text trong Qdrant
                keyword_filter = Filter(
                    must=[
                        FieldCondition(key="group_id", match=MatchValue(value="PUBLIC")),
                        # Tìm keyword trong nội dung (Full Text Match)
                        FieldCondition(key="text", match=MatchText(text=keyword))
                    ]
                )

                # Dùng scroll để lấy các docs chứa keyword
                # Lưu ý: Scroll không sort theo score, nhưng ta gán score = 1.0
                found_docs = qdrant.scroll(
                    collection_name=settings.COLLECTION_NAME,
                    scroll_filter=keyword_filter,
                    limit=2,  # Lấy 2 docs tốt nhất cho mỗi keyword để tránh loãng
                    with_payload=True
                )[0]

                for p in found_docs:
                    sparse_results.append({
                        "id": str(p.id),
                        "score": 1.0,  # Score giả định cho keyword match
                        "payload": p.payload
                    })

            except Exception as e:
                # Thường lỗi do field chưa được đánh index text, bỏ qua để không crash
                logger.warning(f"[HYBRID] Keyword search error for '{keyword}': {e}")

        logger.info(f"[HYBRID] Dense: {len(dense_results)}, Sparse: {len(sparse_results)}")

        # 3. MERGE (RRF)
        merged = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        return merged[:limit]

    except Exception as e:
        logger.error(f"[HYBRID_SEARCH] Error: {e}")
        return []


def reciprocal_rank_fusion(
        dense_results: List[dict],
        sparse_results: List[dict],
        k: int = 60
) -> List[dict]:
    """
    Reciprocal Rank Fusion (Giữ nguyên logic của bạn vì đã đúng)
    """
    scores = {}

    # Score dense results (Ưu tiên Vector hơn: 1.5)
    for rank, doc in enumerate(dense_results, 1):
        doc_id = str(doc.get("id"))
        scores[doc_id] = scores.get(doc_id, 0) + (1.5 / (k + rank))

    # Score sparse results (Keyword match: 1.0)
    for rank, doc in enumerate(sparse_results, 1):
        doc_id = str(doc.get("id"))
        scores[doc_id] = scores.get(doc_id, 0) + (1.0 / (k + rank))

    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Map lại documents
    all_docs = {str(d["id"]): d for d in dense_results + sparse_results}

    merged_results = []
    for doc_id in ranked_ids:
        if doc_id in all_docs:
            doc = all_docs[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            merged_results.append(doc)

    return merged_results

# def save_message(
#         user_id: str,
#         tenancy: str,
#         prompt: str,
#         answer: str,
#         vector: List[float],
#         sources: List[Dict]
# ):
#     """
#     Save user message + AI response vào Qdrant.
#     ✅ Có deduplication check
#     ✅ Lưu vào personal data (group_id=user_id)
#     """
#     try:
#         qdrant = get_client()
#
#         # Create content hash
#         content_for_hash = f"{prompt}::{answer[:500]}"
#         content_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()[:16]
#
#         # Check duplicate
#         try:
#             existing = qdrant.scroll(
#                 collection_name=settings.COLLECTION_NAME,
#                 scroll_filter=Filter(
#                     must=[
#                         FieldCondition(key="group_id", match=MatchValue(value=user_id)),
#                         FieldCondition(key="tenancy", match=MatchValue(value=tenancy)),
#                         FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
#                     ]
#                 ),
#                 limit=1,
#                 with_payload=True
#             )
#
#             if existing[0]:
#                 logger.info(f"[SAVE_MESSAGE] Duplicate detected, skipping. hash={content_hash}")
#                 return
#         except Exception as e:
#             logger.warning(f"[SAVE_MESSAGE] Error checking duplicate: {e}")
#
#         # Save new message
#         qdrant.upsert(
#             collection_name=settings.COLLECTION_NAME,
#             points=[PointStruct(
#                 id=str(uuid.uuid4()),
#                 vector=vector,
#                 payload={
#                     "group_id": user_id,
#                     "tenancy": tenancy,
#                     "prompt": prompt,
#                     "answer": answer,
#                     "sources": sources,
#                     "timestamp": datetime.utcnow().isoformat(),
#                     "content_hash": content_hash,
#                 },
#             )],
#         )
#
#         logger.info(f"[SAVE_MESSAGE] ✓ Saved for user={user_id}, tenancy={tenancy}, hash={content_hash}")
#
#     except Exception as e:
#         logger.error(f"[SAVE_MESSAGE] Error: {e}")
