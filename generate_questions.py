"""
Adversarial question generation for AutoRAGEvals.

Uses Claude to generate 50 adversarial questions across four types:
  - multi_hop       : require connecting info across multiple passages
  - ambiguous       : unclear or debatable answers
  - out_of_scope    : corpus cannot answer
  - paraphrase      : rephrased HotpotQA questions to test robustness

Output is saved to adversarial_questions.json.

Run:
    python generate_questions.py
"""

import json
import os
import random
import textwrap
from typing import List
from dotenv import load_dotenv
import anthropic

from load_data import load_hotpotqa

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-20250514"
OUTPUT_FILE = "adversarial_questions.json"
TOTAL_QUESTIONS = 50
QUESTIONS_PER_TYPE = TOTAL_QUESTIONS // 4          # 12 each, 14 for multi_hop
COUNTS = {
    "multi_hop":   14,
    "ambiguous":   12,
    "out_of_scope": 12,
    "paraphrase":  12,
}
assert sum(COUNTS.values()) == TOTAL_QUESTIONS

# Number of passages / QA pairs to sample and feed to Claude as context
PASSAGE_SAMPLE = 40
QA_SAMPLE = 20
DATA_SPLIT = "validation"
MAX_DATA_LOAD = 300


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_passages(passages: List[str], n: int) -> str:
    sample = random.sample(passages, min(n, len(passages)))
    return "\n\n".join(
        f"[P{i+1}] {p[:600]}" for i, p in enumerate(sample)
    )


def _format_qa_pairs(qa_pairs: List[dict], n: int) -> str:
    sample = random.sample(qa_pairs, min(n, len(qa_pairs)))
    return "\n".join(
        f"- Q: {qa['question']}  A: {qa['answer']}" for qa in sample
    )


SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert at creating adversarial evaluation questions for
    retrieval-augmented generation (RAG) systems. You will be given a sample
    of passages from a corpus and example QA pairs from HotpotQA.

    Your task is to generate evaluation questions that stress-test a RAG
    pipeline. Return ONLY a valid JSON array — no markdown fences, no
    explanation, no preamble. Each element must be a JSON object with
    exactly these fields:
      "type"            : one of multi_hop | ambiguous | out_of_scope | paraphrase
      "question"        : the question string
      "expected_answer" : the answer string, or null for out_of_scope questions
    """)


def _user_prompt(q_type: str, count: int, passages_text: str, qa_text: str) -> str:
    type_instructions = {
        "multi_hop": (
            f"Generate {count} MULTI-HOP questions that require combining "
            "information from at least two different passages to answer. "
            "The answer must be derivable from the corpus."
        ),
        "ambiguous": (
            f"Generate {count} AMBIGUOUS questions whose answers are unclear, "
            "debatable, or could be interpreted in multiple ways given the corpus. "
            "Set expected_answer to the most defensible answer."
        ),
        "out_of_scope": (
            f"Generate {count} OUT-OF-SCOPE questions that cannot be answered "
            "from the corpus at all. Set expected_answer to null."
        ),
        "paraphrase": (
            f"Generate {count} PARAPHRASE questions by rewriting the provided "
            "HotpotQA questions using different wording while preserving meaning. "
            "Keep the same expected_answer as the original."
        ),
    }

    return textwrap.dedent(f"""\
        {type_instructions[q_type]}

        ---PASSAGES---
        {passages_text}

        ---EXAMPLE QA PAIRS (for paraphrase reference)---
        {qa_text}

        Return a JSON array of exactly {count} objects.
        """)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_adversarial_questions(
    passages: List[str],
    qa_pairs: List[dict],
) -> List[dict]:
    """
    Call Claude once per question type and return all questions combined.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    all_questions: List[dict] = []

    for q_type, count in COUNTS.items():
        print(f"  Generating {count} '{q_type}' questions...")

        passages_text = _format_passages(passages, PASSAGE_SAMPLE)
        qa_text = _format_qa_pairs(qa_pairs, QA_SAMPLE)

        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": _user_prompt(q_type, count, passages_text, qa_text),
                }
            ],
        )

        raw = message.content[0].text.strip()

        try:
            questions = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Claude returned invalid JSON for type '{q_type}'.\n"
                f"Raw response:\n{raw}"
            ) from exc

        if not isinstance(questions, list):
            raise ValueError(
                f"Expected a JSON array for type '{q_type}', got: {type(questions)}"
            )

        # Enforce correct type field and required keys
        for q in questions:
            q["type"] = q_type
            if "expected_answer" not in q:
                q["expected_answer"] = None

        all_questions.extend(questions)
        print(f"    -> {len(questions)} questions received.")

    return all_questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading HotpotQA data...")
    passages, qa_pairs = load_hotpotqa(
        split=DATA_SPLIT,
        max_samples=MAX_DATA_LOAD,
    )
    print(f"  {len(passages)} passages, {len(qa_pairs)} QA pairs loaded.")

    print(f"\nGenerating {TOTAL_QUESTIONS} adversarial questions with {MODEL}...")
    questions = generate_adversarial_questions(passages, qa_pairs)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(questions)} questions to '{OUTPUT_FILE}'.")

    # Print a sample
    print("\nSample (one per type):")
    seen = set()
    for q in questions:
        if q["type"] not in seen:
            seen.add(q["type"])
            print(f"  [{q['type']}] {q['question']}")
            print(f"           -> {q['expected_answer']}")


if __name__ == "__main__":
    main()
