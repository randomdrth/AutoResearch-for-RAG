"""
Entry point for the AutoRAGEvals optimizer.

Run:
    python run_optimizer.py
"""

import json
from dotenv import load_dotenv

from load_data import load_hotpotqa
from optimizer import (
    ADVERSARIAL_QUESTIONS_FILE,
    DEFAULT_CONFIG,
    EXPERIMENT_LOG_FILE,
    N_SCORING_RUNS,
    run_optimizer,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_SPLIT = "validation"
MAX_DATA_LOAD = 300   # passages to index each iteration


def main() -> None:
    # Load adversarial questions, excluding out_of_scope (no ground truth)
    with open(ADVERSARIAL_QUESTIONS_FILE, encoding="utf-8") as f:
        all_questions = json.load(f)
    questions = [q for q in all_questions if q.get("expected_answer")]
    print(f"Loaded {len(questions)} scoreable adversarial questions "
          f"(excluded {len(all_questions) - len(questions)} out_of_scope).")

    # Load passages for re-indexing
    print(f"Loading HotpotQA passages ({DATA_SPLIT}, max {MAX_DATA_LOAD})...")
    passages, _ = load_hotpotqa(split=DATA_SPLIT, max_samples=MAX_DATA_LOAD)
    print(f"  {len(passages)} passages ready.\n")

    print(f"Scoring each config {N_SCORING_RUNS} times (mean ± std).\n")

    result = run_optimizer(questions, passages, log_file=EXPERIMENT_LOG_FILE)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    baseline     = result["baseline_scores"]
    best         = result["best_scores"]
    best_std     = result["best_scores_std"]

    print("\n" + "=" * 76)
    print("OPTIMIZATION COMPLETE")
    print("=" * 76)
    print(f"\nBaseline config : {result['baseline_config']}")
    print(f"Best config     : {result['best_config']}")
    print(f"Total experiments run: {result['total_iterations']}")
    print(f"Scoring runs per config: {N_SCORING_RUNS}\n")

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "overall",
        "rouge_l",
    ]
    header = (
        f"{'Metric':<25} {'Baseline':>10} {'Best mean':>10} "
        f"{'Best std':>10} {'Delta':>10}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        b   = baseline.get(m, 0.0)
        bst = best.get(m, 0.0)
        std = best_std.get(m, 0.0)
        print(
            f"{m:<25} {b:>10.4f} {bst:>10.4f} "
            f"±{std:>8.4f} {bst - b:>+10.4f}"
        )


if __name__ == "__main__":
    main()
