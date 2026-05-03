"""
RAGAS scorer for AutoRAGEvals.

Usage
-----
    from scorer import score_results

    results = [
        {
            "question"    : "Who wrote Hamlet?",
            "answer"      : "William Shakespeare wrote Hamlet.",
            "contexts"    : ["William Shakespeare was an English playwright..."],
            "ground_truth": "William Shakespeare",
        },
        ...
    ]

    scores = score_results(results)
    # {"faithfulness": 0.95, "answer_relevancy": 0.88, ..., "overall": 0.91}
"""

import os
import concurrent.futures
from typing import List, Dict, Any

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

JUDGE_MODEL = "gpt-4o-mini"
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def score_results(
    results: List[Dict[str, Any]],
    judge_model: str = JUDGE_MODEL,
) -> Dict[str, float]:
    """
    Run RAGAS evaluation on a list of RAG results.

    Parameters
    ----------
    results : list of dict
        Each dict must contain:
            - ``question``     : str
            - ``answer``       : str   (RAG-generated answer)
            - ``contexts``     : list[str]  (retrieved chunks)
            - ``ground_truth`` : str   (gold answer)
    judge_model : str
        OpenAI model used by RAGAS as the judge LLM.

    Returns
    -------
    dict
        Keys: faithfulness, answer_relevancy, context_precision,
              context_recall, overall (average of the four).
    """
    if not results:
        raise ValueError("score_results() received an empty list — nothing to evaluate.")

    # Build HuggingFace Dataset in the shape RAGAS expects
    dataset = Dataset.from_dict(
        {
            "question":     [r["question"] for r in results],
            "answer":       [r["answer"] for r in results],
            "contexts":     [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }
    )

    llm = ChatOpenAI(
        model=judge_model,
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # Run in a separate thread to avoid event-loop conflicts with LlamaIndex.
    #
    # Python 3.9 caveat: asyncio.Semaphore binds to the *current* loop at
    # creation time, but RAGAS internally calls asyncio.run() which always
    # creates a *new* loop — causing a "Future attached to a different loop"
    # error.  Replacing asyncio.run with loop.run_until_complete for the
    # duration of the call keeps the Semaphore and the runner on the same loop.
    def _run() -> Any:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        original_run = asyncio.run
        asyncio.run = loop.run_until_complete  # type: ignore[assignment]
        try:
            return evaluate(dataset=dataset, metrics=METRICS, llm=llm)
        finally:
            asyncio.run = original_run  # type: ignore[assignment]
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        ragas_result = pool.submit(_run).result()

    scores: Dict[str, float] = {}
    for name in METRIC_NAMES:
        val = ragas_result[name]
        scores[name] = round(float(val), 4)

    scores["overall"] = round(sum(scores.values()) / len(scores), 4)
    return scores
