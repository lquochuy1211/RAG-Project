import json
import sys
import os

# Thêm đường dẫn root để import được module app
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.qdrant_client import search_similar_hybrid
from app.data_ingestion.data_processor import DataProcessor

INPUT_FILE = "scripts/evaluation/golden_dataset.json"


def calculate_hit_rate(dataset, top_k=5):
    processor = DataProcessor()  # Load model embedding
    hits = 0
    total = len(dataset)

    print(f"Running evaluation on {total} queries (Top-K={top_k})...")

    for idx, item in enumerate(dataset):
        query = item["question"]
        target_id = item["ground_truth_id"]

        # 1. Tạo vector query
        query_vector = processor.embedder.encode(query).tolist()

        # 2. Gọi hàm search thực tế của hệ thống
        # Lưu ý: search_similar_hybrid trả về list dict có key 'id'
        results = search_similar_hybrid(
            query_vector=query_vector,
            query_text=query,
            limit=top_k
        )

        # 3. Kiểm tra xem target_id có trong list results không
        found = False
        for res in results:
            if str(res["id"]) == target_id:
                found = True
                break

        if found:
            hits += 1
            print(f"[{idx + 1}/{total}] ✅ Found: {query}")
        else:
            print(f"[{idx + 1}/{total}] ❌ Missed: {query}")

    hit_rate = hits / total
    return hit_rate


if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    score = calculate_hit_rate(data, top_k=5)
    print("=" * 40)
    print(f"FINAL HIT RATE @ 5: {score * 100:.2f}%")
    print("=" * 40)