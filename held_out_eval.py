"""
Held-out evaluation for AutoRAGEvals.

Runs all 50 questions from hotpot_heldout_50.json through the RAG pipeline
under two configs, scores each with RAGAS + ROUGE-L, and prints a comparison
table.  Results are saved to held_out_results.json.

Run:
    python held_out_eval.py
"""

import json
import os
from datetime import datetime
from typing import Dict, List

import chromadb
from dotenv import load_dotenv
from rouge_score import rouge_scorer as rouge_scorer_lib

from load_data import load_hotpotqa
from pipeline import RAGPipeline
from scorer import score_results

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HELDOUT_FILE     = "hotpot_heldout_50.json"
RESULTS_FILE     = "held_out_results.json"
CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "heldout_eval"
DATA_SPLIT       = "validation"
MAX_DATA_LOAD    = 300

BASELINE_CONFIG = {
    "chunk_size":     512,
    "chunk_overlap":  50,
    "top_k":          3,
    "prompt_variant": "baseline",
    "reranker":       False,
}

BEST_CONFIG = {
    "chunk_size":     512,
    "chunk_overlap":  50,
    "top_k":          7,
    "prompt_variant": "baseline",
    "reranker":       False,
}

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "overall",
    "rouge_l",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
        if ref
    ]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def run_config(
    pipeline: RAGPipeline,
    questions: List[Dict],
    top_k: int,
    prompt_variant: str,
    label: str,
) -> Dict:
    """
    Query all questions with the given top_k / prompt_variant, score, return
    a dict with per-question results and aggregate RAGAS + ROUGE-L scores.
    """
    print(f"\n  Running {label} (top_k={top_k}, prompt={prompt_variant})...")

    ragas_inputs = []
    predictions  = []
    references   = []
    per_question = []

    for i, q in enumerate(questions, 1):
        result = pipeline.query(
            q["question"],
            top_k=top_k,
            prompt_variant=prompt_variant,
            use_reranker=False,
        )
        ragas_inputs.append({
            "question":     q["question"],
            "answer":       result["answer"],
            "contexts":     result["contexts"],
            "ground_truth": q["expected_answer"],
        })
        predictions.append(result["answer"])
        references.append(q["expected_answer"])
        per_question.append({
            "hotpot_id":      q.get("hotpot_id", ""),
            "question":       q["question"],
            "expected_answer": q["expected_answer"],
            "answer":         result["answer"],
            "contexts":       result["contexts"],
        })
        if i % 10 == 0:
            print(f"    {i}/{len(questions)} questions queried...")

    print(f"    Scoring with RAGAS...")
    ragas_scores = score_results(ragas_inputs)
    rouge_l      = _compute_rouge_l(predictions, references)

    scores = {**ragas_scores, "rouge_l": rouge_l}
    print(
        f"    overall={scores['overall']:.4f}  "
        f"rouge_l={scores['rouge_l']:.4f}"
    )

    return {"scores": scores, "per_question": per_question}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load held-out questions
    with open(HELDOUT_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} held-out questions from '{HELDOUT_FILE}'.")

    # Load passages
    print(f"Loading HotpotQA passages ({DATA_SPLIT}, max {MAX_DATA_LOAD})...")
    passages, _ = load_hotpotqa(split=DATA_SPLIT, max_samples=MAX_DATA_LOAD)
    print(f"  {len(passages)} passages loaded.")

    # Both configs share the same chunk_size and chunk_overlap, so we build
    # the index once and reuse it for both runs.
    print(f"\nBuilding index (chunk_size={BASELINE_CONFIG['chunk_size']}, "
          f"chunk_overlap={BASELINE_CONFIG['chunk_overlap']})...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    pipeline = RAGPipeline(
        chunk_size=BASELINE_CONFIG["chunk_size"],
        chunk_overlap=BASELINE_CONFIG["chunk_overlap"],
        collection_name=COLLECTION_NAME,
        chroma_path=CHROMA_PATH,
    )
    pipeline.load_documents(passages)
    print("  Index built.")

    # Run both configs
    baseline_result = run_config(
        pipeline, questions,
        top_k=BASELINE_CONFIG["top_k"],
        prompt_variant=BASELINE_CONFIG["prompt_variant"],
        label="BASELINE",
    )
    best_result = run_config(
        pipeline, questions,
        top_k=BEST_CONFIG["top_k"],
        prompt_variant=BEST_CONFIG["prompt_variant"],
        label="BEST",
    )

    baseline_scores = baseline_result["scores"]
    best_scores     = best_result["scores"]

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("HELD-OUT EVALUATION RESULTS")
    print("=" * 68)
    print(f"{'Metric':<25} {'Baseline':>10} {'Best':>10} {'Delta':>10}")
    print("-" * 57)
    for m in METRICS:
        b   = baseline_scores.get(m, 0.0)
        bst = best_scores.get(m, 0.0)
        print(f"{m:<25} {b:>10.4f} {bst:>10.4f} {bst - b:>+10.4f}")
    print("=" * 68)
    print(f"\nBaseline config : {BASELINE_CONFIG}")
    print(f"Best config     : {BEST_CONFIG}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "timestamp":       datetime.utcnow().isoformat(),
        "heldout_file":    HELDOUT_FILE,
        "n_questions":     len(questions),
        "passages_loaded": len(passages),
        "baseline": {
            "config":       BASELINE_CONFIG,
            "scores":       baseline_scores,
            "per_question": baseline_result["per_question"],
        },
        "best": {
            "config":       BEST_CONFIG,
            "scores":       best_scores,
            "per_question": best_result["per_question"],
        },
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to '{RESULTS_FILE}'.")


if __name__ == "__main__":
    main()
