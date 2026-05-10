import json
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from datasets import Dataset

# LangChain wrapper for Perplexity
from langchain_community.chat_models import ChatOpenAI

# --- CONFIG PERPLEXITY FOR RAGAS ---
# Ragas uses LangChain under the hood. We configure it to use Perplexity.
perplexity_llm = ChatOpenAI(
    openai_api_key=os.getenv("PERPLEXITY_API_KEY"),
    openai_api_base="https://api.perplexity.ai",
    model_name="llama-3.1-sonar-large-128k-chat",
    temperature=0
)


# Function to simulate your actual Chatbot Pipeline
def call_my_chatbot(question):
    """
    This function mimics your FastAPI /ask endpoint.
    In a real script, you might request your running FastAPI server:
    requests.post("http://localhost:8000/ask", json={"question": question})
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )

    # 1. (Simulated) Retrieve context manually for the test
    # In reality, your /ask endpoint does this internally.
    # Here we just ask Perplexity to answer based on general knowledge
    # OR you should plug in your real retrieval logic here.

    response = client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",  # Use the ONLINE model for fresh data
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Vietnam History and Tourism."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


def main():
    # 1. Load the Golden Dataset created in Step 1
    with open("scripts/evaluation/golden_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 2. Prepare Data
    data_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"Running evaluation on {len(raw_data)} samples...")

    for idx, item in enumerate(raw_data):
        question = item["question"]
        print(f"[{idx + 1}] Asking: {question}")

        # Call your bot (using Perplexity)
        bot_answer = call_my_chatbot(question)

        # We assume the bot 'found' the correct context for the sake of this test
        # (In a strict test, you would fetch what the bot actually retrieved)
        retrieved_contexts = [item["ground_truth_context"]]

        data_dict["question"].append(question)
        data_dict["answer"].append(bot_answer)
        data_dict["contexts"].append(retrieved_contexts)
        data_dict["ground_truth"].append(item["ground_truth_context"])

    # 3. Configure Ragas to use Perplexity as the Judge
    # Note: Ragas works best with GPT-4, but Perplexity (Llama 3) is a decent substitute.
    dataset = Dataset.from_dict(data_dict)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevance, context_precision],
        llm=perplexity_llm,  # <--- Pass Perplexity here
        embeddings=None  # Uses OpenAI default, or pass your own if needed
    )

    print("\nEvaluation Results:")
    print(results)

    # Save to CSV for your report
    df = results.to_pandas()
    df.to_csv("scripts/evaluation/ragas_report_perplexity.csv")
    print("Saved detailed report to scripts/evaluation/ragas_report_perplexity.csv")


if __name__ == "__main__":
    main()