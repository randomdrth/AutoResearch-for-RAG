"""
AutoRAGEvals optimizer loop.

Uses Claude to propose config changes one at a time, rebuilds the RAG
pipeline, scores with RAGAS (keep/revert criterion) and ROUGE-L (logged
only), and persists every experiment to experiment_log.json.

Resumable: if the log file already exists the loop picks up where it
left off, restoring the current config from the last kept entry.

Note on retrieval mode:
    Hybrid (dense + BM25) was evaluated but skipped — it requires
    llama-index-retrievers-bm25 + rank_bm25, which introduce additional
    pip resolver conflicts on Python 3.9.  The search space covers
    chunk_size, chunk_overlap, and top_k only.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import chromadb
from dotenv import load_dotenv
from rouge_score import rouge_scorer as rouge_scorer_lib

from load_data import load_hotpotqa
from pipeline import RAGPipeline
from scorer import score_results

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ADVERSARIAL_QUESTIONS_FILE = "adversarial_questions.json"
EXPERIMENT_LOG_FILE = "experiment_log.json"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CHROMA_PATH = "./chroma_db"
OPTIMIZER_COLLECTION = "optimizer_temp"

MAX_ITERATIONS = 15
NO_IMPROVE_LIMIT = 3
PROPOSAL_RETRIES = 3  # max Claude retries if proposal is invalid

DEFAULT_CONFIG: Dict[str, Any] = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 3,
}

# Only values listed here are valid proposals from Claude.
SEARCH_SPACE: Dict[str, List[Any]] = {
    "chunk_size": [256, 512, 1024],
    "chunk_overlap": [25, 50, 100],
    "top_k": [3, 5, 7],
}


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
        if ref  # skip empty references
    ]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_on_questions(
    questions: List[Dict],
    config: Dict[str, Any],
    passages: List[str],
    collection_name: str = OPTIMIZER_COLLECTION,
    chroma_path: str = CHROMA_PATH,
) -> Dict[str, float]:
    """
    Build a fresh pipeline with *config*, run every question, return
    RAGAS metrics + rouge_l.

    out_of_scope questions (expected_answer is None/empty) are excluded
    from both RAGAS and ROUGE-L since there is no ground truth.
    """
    scoreable = [q for q in questions if q.get("expected_answer")]

    # Wipe the temp collection so stale embeddings from the prior config
    # don't pollute results.
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    pipeline = RAGPipeline(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        collection_name=collection_name,
        chroma_path=chroma_path,
    )
    pipeline.load_documents(passages)

    ragas_inputs: List[Dict] = []
    predictions: List[str] = []
    references: List[str] = []

    for q in scoreable:
        result = pipeline.query(q["question"], top_k=config["top_k"])
        ragas_inputs.append({
            "question":     q["question"],
            "answer":       result["answer"],
            "contexts":     result["contexts"],
            "ground_truth": q["expected_answer"],
        })
        predictions.append(result["answer"])
        references.append(q["expected_answer"])

    ragas_scores = score_results(ragas_inputs)
    rouge_l = _compute_rouge_l(predictions, references)

    return {**ragas_scores, "rouge_l": rouge_l}


# ---------------------------------------------------------------------------
# Claude proposal
# ---------------------------------------------------------------------------

def _propose_config_change(
    current_config: Dict[str, Any],
    current_scores: Dict[str, float],
    history: List[Dict],
    client: anthropic.Anthropic,
) -> Dict[str, Any]:
    """
    Ask Claude to propose the next single-parameter change.
    Returns a dict: {parameter, old_value, new_value, reasoning}.
    Raises ValueError if Claude returns invalid JSON after PROPOSAL_RETRIES.
    """
    # Summarise recent history for the prompt (cap at 12 to stay compact)
    history_summary = [
        {
            "config":   e["config"],
            "overall":  e["scores"]["overall"],
            "kept":     e["kept"],
            "reasoning": e.get("reasoning", ""),
        }
        for e in history[-12:]
    ]

    # Configs already tried (and rejected) — Claude should avoid these
    tried_configs = [
        e["config"] for e in history
        if e.get("type") == "experiment" and not e["kept"]
    ]

    prompt = f"""You are optimizing a RAG pipeline by adjusting one hyperparameter at a time.
Goal: maximise the overall RAGAS score.

Current config:
{json.dumps(current_config, indent=2)}

Current scores (higher is better, range 0-1):
{json.dumps({k: v for k, v in current_scores.items() if k != "rouge_l"}, indent=2)}

Valid search space (only these values are allowed):
{json.dumps(SEARCH_SPACE, indent=2)}

Recent experiment history:
{json.dumps(history_summary, indent=2)}

Configs already tried and rejected (do not propose these):
{json.dumps(tried_configs, indent=2)}

Rules:
- Propose exactly ONE parameter change.
- The new_value must come from the search space for that parameter.
- Do not propose a config that appears in the rejected list above.
- Prefer parameters where the score trend suggests room for improvement.

Return ONLY a valid JSON object — no markdown, no preamble — with exactly these fields:
  "parameter"  : one of {list(SEARCH_SPACE.keys())}
  "old_value"  : the current value of that parameter
  "new_value"  : the proposed new value (must be from the search space)
  "reasoning"  : one sentence explaining the choice
"""

    last_error: Optional[str] = None
    for attempt in range(PROPOSAL_RETRIES):
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        try:
            proposal = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error: {exc}\nRaw: {raw}"
            continue

        # Validate fields
        param = proposal.get("parameter")
        new_val = proposal.get("new_value")

        if param not in SEARCH_SPACE:
            last_error = f"Unknown parameter: {param}"
            continue
        if new_val not in SEARCH_SPACE[param]:
            last_error = f"Value {new_val} not in search space for {param}"
            continue
        if new_val == current_config.get(param):
            last_error = f"Proposed value equals current value for {param}"
            continue

        return proposal

    raise ValueError(
        f"Claude failed to produce a valid proposal after {PROPOSAL_RETRIES} "
        f"attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _save_log(log: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def _load_log(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def run_optimizer(
    questions: List[Dict],
    passages: List[str],
    max_iterations: int = MAX_ITERATIONS,
    no_improve_limit: int = NO_IMPROVE_LIMIT,
    log_file: str = EXPERIMENT_LOG_FILE,
) -> Dict[str, Any]:
    """
    Run the optimization loop.

    Parameters
    ----------
    questions : list of dicts with keys: question, expected_answer, type
    passages  : text passages to index (re-indexed each iteration)
    max_iterations   : hard cap on number of experiments
    no_improve_limit : stop after this many consecutive non-improvements
    log_file  : path to the JSON experiment log (enables resumption)

    Returns
    -------
    dict with keys: baseline_config, baseline_scores, best_config,
                    best_scores, total_iterations, log
    """
    anthropic_client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    # ------------------------------------------------------------------
    # Resume or start fresh
    # ------------------------------------------------------------------
    if os.path.exists(log_file):
        log = _load_log(log_file)
        print(f"Resuming from '{log_file}' ({len(log)} entries already logged).")

        kept_entries = [e for e in log if e["kept"]]
        # current = last kept entry (baseline is always kept, so this is safe)
        current_config = kept_entries[-1]["config"].copy()
        current_scores = kept_entries[-1]["scores"].copy()

        # Count consecutive non-improvements at the tail of the log
        no_improve_streak = 0
        for entry in reversed(log):
            if entry.get("type") == "baseline":
                break
            if not entry["kept"]:
                no_improve_streak += 1
            else:
                break

        last_iter = max(e["iteration"] for e in log)
    else:
        log = []
        current_config = DEFAULT_CONFIG.copy()
        current_scores = {}  # will be filled by baseline scoring below
        no_improve_streak = 0
        last_iter = -1

    # ------------------------------------------------------------------
    # Baseline (only if not already in log)
    # ------------------------------------------------------------------
    if not any(e.get("type") == "baseline" for e in log):
        print("Scoring baseline config...")
        current_scores = score_on_questions(questions, current_config, passages)
        log.append({
            "iteration": 0,
            "type":      "baseline",
            "config":    current_config.copy(),
            "scores":    current_scores,
            "kept":      True,
            "reasoning": "Baseline — starting point.",
            "timestamp": datetime.utcnow().isoformat(),
        })
        _save_log(log, log_file)
        last_iter = 0
        print(
            f"  Baseline  overall={current_scores['overall']:.4f}"
            f"  rouge_l={current_scores['rouge_l']:.4f}"
        )

    baseline_scores = next(e["scores"] for e in log if e.get("type") == "baseline")

    # Track best across all kept entries
    best_entry = max((e for e in log if e["kept"]), key=lambda e: e["scores"]["overall"])
    best_config: Dict[str, Any] = best_entry["config"].copy()
    best_overall: float = best_entry["scores"]["overall"]

    # ------------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------------
    for iteration in range(last_iter + 1, max_iterations + 1):
        if no_improve_streak >= no_improve_limit:
            print(
                f"\nStopping: no improvement for {no_improve_limit} "
                "consecutive iterations."
            )
            break

        print(f"\n--- Iteration {iteration} / {max_iterations} ---")
        print(f"  Current config : {current_config}")
        print(f"  Current overall: {current_scores['overall']:.4f}")

        # Ask Claude for a proposal
        try:
            proposal = _propose_config_change(
                current_config, current_scores, log, anthropic_client
            )
        except ValueError as exc:
            print(f"  Proposal failed: {exc}  Stopping.")
            break

        param     = proposal["parameter"]
        new_val   = proposal["new_value"]
        reasoning = proposal["reasoning"]
        print(f"  Proposal : {param}  {proposal['old_value']} → {new_val}")
        print(f"  Reasoning: {reasoning}")

        candidate_config = {**current_config, param: new_val}

        # Score candidate
        print("  Scoring candidate (re-indexing passages)...")
        candidate_scores = score_on_questions(questions, candidate_config, passages)

        kept = candidate_scores["overall"] > current_scores["overall"]
        verdict = "KEPT" if kept else "REVERTED"
        delta = candidate_scores["overall"] - current_scores["overall"]
        print(
            f"  Result : overall={candidate_scores['overall']:.4f}"
            f"  rouge_l={candidate_scores['rouge_l']:.4f}"
            f"  Δ={delta:+.4f}  [{verdict}]"
        )

        log.append({
            "iteration": iteration,
            "type":      "experiment",
            "proposal":  proposal,
            "config":    candidate_config,
            "scores":    candidate_scores,
            "kept":      kept,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat(),
        })
        _save_log(log, log_file)

        if kept:
            current_config = candidate_config
            current_scores = candidate_scores
            no_improve_streak = 0
            if current_scores["overall"] > best_overall:
                best_overall = current_scores["overall"]
                best_config  = current_config.copy()
        else:
            no_improve_streak += 1

    # ------------------------------------------------------------------
    # Return summary
    # ------------------------------------------------------------------
    best_scores = max(
        (e["scores"] for e in log if e["kept"]),
        key=lambda s: s["overall"],
    )

    return {
        "baseline_config":  DEFAULT_CONFIG,
        "baseline_scores":  baseline_scores,
        "best_config":      best_config,
        "best_scores":      best_scores,
        "total_iterations": sum(1 for e in log if e.get("type") == "experiment"),
        "log":              log,
    }
