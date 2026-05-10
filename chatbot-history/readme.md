1. Cài dependencies:
   pip install -r requirements.txt

2. Thiết lập .env (xem .env example)

3. Chạy app:
   uvicorn app.main:app --reload

4. Endpoint:
   POST /ask/  (body JSON)
   {
     "prompt": "Lịch sử Đấu trường La Mã và giờ thăm quan hôm nay?",
     "user_id": "user123",
     "top_k": 5,
     "use_keyword": false
   }

5. Lưu ý và hướng phát triển tiếp:
   - Triển khai sentence-transformers (all-MiniLM-L6-v2) nếu muốn embedding cục bộ. Set EMBEDDING_PROVIDER=SENTENCE_TRANSFORMERS và cài sentence-transformers.
   - Thay Perplexity bằng Llama 3 (self-hosted) hoặc OpenAI Chat (gọi ChatCompletions) nếu muốn kiểm soát tốt hơn.
   - Thực hiện entity-extraction (spaCy hoặc OpenAI) để dùng sparse retrieval thật (keyword matches).
   - Thực hiện RRF (Reciprocal Rank Fusion) nếu tích hợp nhiều nguồn retrieval.
   - Thêm rate limit, auth, caching, monitoring.
