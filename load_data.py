"""
Download HotpotQA (distractor split) from HuggingFace and extract
context passages + QA pairs into clean Python structures.

Usage
-----
    from load_data import load_hotpotqa

    passages, qa_pairs = load_hotpotqa(split="train", max_samples=500)

    # passages  -> list[str]  (deduplicated context paragraphs)
    # qa_pairs  -> list[dict] with keys: question, answer, supporting_titles
"""

from typing import Optional, Tuple, List
from datasets import load_dataset


def load_hotpotqa(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[dict]]:
    """
    Load HotpotQA (distractor config) and return passages and QA pairs.

    Parameters
    ----------
    split : str
        HuggingFace dataset split — "train", "validation".
        (The test split has no answers, so it is not useful here.)
    max_samples : int, optional
        Truncate to this many examples. ``None`` loads the full split.

    Returns
    -------
    passages : list[str]
        Deduplicated context paragraphs ready to feed into the pipeline.
    qa_pairs : list[dict]
        Each dict contains:
            - ``question``          : str
            - ``answer``            : str
            - ``supporting_titles`` : list[str]
    """
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    seen: set[str] = set()
    passages: list[str] = []
    qa_pairs: list[dict] = []

    for example in dataset:
        # --- context passages -------------------------------------------
        # Each example has a "context" field: {"title": [...], "sentences": [[...]]}
        titles = example["context"]["title"]
        sentences_per_doc = example["context"]["sentences"]

        for title, sentences in zip(titles, sentences_per_doc):
            paragraph = " ".join(sentences).strip()
            key = f"{title}|||{paragraph[:120]}"  # dedup key
            if key not in seen:
                seen.add(key)
                passages.append(paragraph)

        # --- QA pair ----------------------------------------------------
        qa_pairs.append(
            {
                "question": example["question"],
                "answer": example["answer"],
                "supporting_titles": example["supporting_facts"]["title"],
            }
        )

    return passages, qa_pairs


if __name__ == "__main__":
    print("Loading HotpotQA validation split (first 50 examples)...")
    passages, qa_pairs = load_hotpotqa(split="validation", max_samples=50)
    print(f"  Passages : {len(passages)}")
    print(f"  QA pairs : {len(qa_pairs)}")
    print("\nSample QA pair:")
    sample = qa_pairs[0]
    print(f"  Q: {sample['question']}")
    print(f"  A: {sample['answer']}")
    print(f"  Supporting docs: {sample['supporting_titles']}")
    print("\nSample passage (truncated):")
    print(f"  {passages[0][:200]}...")
