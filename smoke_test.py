"""
Smoke test: load 5 HotpotQA questions, run them through the RAG pipeline,
and print question / answer / retrieved contexts for each.

Run:
    python smoke_test.py
"""

from load_data import load_hotpotqa
from pipeline import RAGPipeline
from scorer import score_results

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_QA_SAMPLES = 5          # number of QA pairs to test
DATA_SPLIT = "validation"   # use validation so answers are available
MAX_DATA_LOAD = 200         # load enough examples to get NUM_QA_SAMPLES
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3


def main() -> None:
    # 1. Load data
    print("=" * 70)
    print("Loading HotpotQA data...")
    passages, qa_pairs = load_hotpotqa(
        split=DATA_SPLIT,
        max_samples=MAX_DATA_LOAD,
    )
    print(f"  {len(passages)} context passages loaded")
    print(f"  {len(qa_pairs)} QA pairs loaded")

    # 2. Build pipeline and ingest passages
    print("\nBuilding RAG pipeline and ingesting passages...")
    pipeline = RAGPipeline(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    pipeline.load_documents(passages)
    print("  Ingestion complete.")

    # 3. Run smoke test queries
    print("\n" + "=" * 70)
    print(f"Running {NUM_QA_SAMPLES} smoke-test queries (top_k={TOP_K})")
    print("=" * 70)

    ragas_inputs = []

    for i, qa in enumerate(qa_pairs[:NUM_QA_SAMPLES], start=1):
        question = qa["question"]
        gold_answer = qa["answer"]

        result = pipeline.query(question, top_k=TOP_K)

        print(f"\n--- Query {i} ---")
        print(f"Question : {question}")
        print(f"Gold ans : {gold_answer}")
        print(f"RAG ans  : {result['answer']}")
        print(f"Retrieved contexts ({len(result['contexts'])}):")
        for j, ctx in enumerate(result["contexts"], start=1):
            snippet = ctx[:200].replace("\n", " ")
            print(f"  [{j}] {snippet}{'...' if len(ctx) > 200 else ''}")

        ragas_inputs.append(
            {
                "question":     question,
                "answer":       result["answer"],
                "contexts":     result["contexts"],
                "ground_truth": gold_answer,
            }
        )

    # 4. RAGAS scoring
    print("\n" + "=" * 70)
    print("Running RAGAS evaluation...")
    scores = score_results(ragas_inputs)

    print("\nRAGAS scores:")
    for metric, val in scores.items():
        label = f"  {metric:<22}"
        print(f"{label}: {val:.4f}")

    print("\n" + "=" * 70)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
