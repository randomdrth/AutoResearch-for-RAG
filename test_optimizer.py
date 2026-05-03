"""
Smoke test for the optimizer: 2 iterations on 10 adversarial questions.

This is intentionally fast and cheap — use it to confirm the optimizer
loop works end-to-end before running the full run_optimizer.py.

Run:
    python test_optimizer.py
"""

import json
import os
from dotenv import load_dotenv

from load_data import load_hotpotqa
from optimizer import run_optimizer

load_dotenv()

# ---------------------------------------------------------------------------
# Smoke-test config
# ---------------------------------------------------------------------------
TEST_LOG_FILE      = "test_experiment_log.json"
TEST_N_QUESTIONS   = 10
TEST_MAX_ITERATIONS = 2
TEST_MAX_DATA_LOAD  = 100   # smaller corpus → faster re-indexing


def main() -> None:
    # Load adversarial questions (first N scoreable ones)
    with open("adversarial_questions.json", encoding="utf-8") as f:
        all_questions = json.load(f)
    questions = [q for q in all_questions if q.get("expected_answer")]
    questions = questions[:TEST_N_QUESTIONS]
    print(f"Smoke test: {len(questions)} questions, "
          f"{TEST_MAX_ITERATIONS} iterations max.\n")

    # Load a small passage corpus
    print(f"Loading HotpotQA passages (max {TEST_MAX_DATA_LOAD})...")
    passages, _ = load_hotpotqa(split="validation", max_samples=TEST_MAX_DATA_LOAD)
    print(f"  {len(passages)} passages loaded.\n")

    # Always start fresh for the smoke test
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)
        print(f"Removed previous '{TEST_LOG_FILE}'.\n")

    result = run_optimizer(
        questions=questions,
        passages=passages,
        max_iterations=TEST_MAX_ITERATIONS,
        no_improve_limit=3,
        log_file=TEST_LOG_FILE,
    )

    # ------------------------------------------------------------------
    # Print full experiment log
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT LOG")
    print("=" * 70)

    for entry in result["log"]:
        itr     = entry["iteration"]
        cfg     = entry["config"]
        scores  = entry["scores"]
        kept    = entry["kept"]
        label   = entry.get("type", "experiment").upper()
        reason  = entry.get("reasoning", "—")

        print(f"\n[{label} | iter={itr}]")
        print(f"  config  : {cfg}")
        print(f"  overall : {scores['overall']:.4f}   rouge_l : {scores['rouge_l']:.4f}")
        print(f"  faithfulness={scores['faithfulness']:.4f}  "
              f"answer_relevancy={scores['answer_relevancy']:.4f}  "
              f"context_precision={scores['context_precision']:.4f}  "
              f"context_recall={scores['context_recall']:.4f}")
        print(f"  kept    : {kept}")
        print(f"  reason  : {reason}")

    print("\n" + "=" * 70)
    print(f"Best config : {result['best_config']}")
    print(f"Best overall: {result['best_scores']['overall']:.4f}")
    print(f"Baseline    : {result['baseline_scores']['overall']:.4f}")
    delta = result['best_scores']['overall'] - result['baseline_scores']['overall']
    print(f"Delta       : {delta:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
