import json
import os
from qdrant_client import QdrantClient
from openai import OpenAI  # We still use this SDK, but configured for Perplexity
from app.config.settings import settings

# --- CONFIG PERPLEXITY ---
PERPLEXITY_API_KEY = os.getenv("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")  # Ensure this is set
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
PERPLEXITY_MODEL = "llama-3.1-sonar-large-128k-chat"  # Good for reasoning

# Init Clients
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
pplx_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url=PERPLEXITY_BASE_URL)

QDRANT_COLLECTION = settings.COLLECTION_NAME
OUTPUT_FILE = "scripts/evaluation/golden_dataset.json"
NUM_SAMPLES = 50


def generate_question(context: str) -> str:
    """Use Perplexity to generate a question from the context."""
    prompt = f"""
    You are an expert Vietnamese historian and tourism guide.
    Based strictly on the text below, generate ONE specific question that this text answers.

    Rules:
    - The question must be in Vietnamese.
    - Do not answer the question.
    - Output ONLY the question.

    Text: {context[:1500]}...
    """

    response = pplx_client.chat.completions.create(
        model=PERPLEXITY_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def main():
    print("1. Fetching random chunks from Qdrant...")
    points_response = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=NUM_SAMPLES * 2,
        with_payload=True
    )[0]

    dataset = []
    print(f"2. generating questions using Perplexity ({PERPLEXITY_MODEL})...")

    for point in points_response:
        if len(dataset) >= NUM_SAMPLES:
            break

        text = point.payload.get("text", "")
        if len(text) < 150: continue

        try:
            question = generate_question(text)
            dataset.append({
                "id": str(point.id),
                "question": question,
                "ground_truth_context": text,
                "ground_truth_id": str(point.id)
            })
            print(f"   [OK] {question}")
        except Exception as e:
            print(f"   [ERR] {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Saved {len(dataset)} pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()