"""
Evaluation harness using RAGAS metrics.

This is what turns "I built a RAG app" into "I built a RAG app and measured
a 23% -> 4% hallucination reduction from self-correction" — the second one
is what actually lands in an interview. Build an eval_dataset.json of
~20-30 question/ground-truth pairs covering simple, multi-hop, and
comparison questions, then run this twice: once with self-correction and
citation verification disabled (baseline), once with them enabled.
"""
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (answer_relevancy, context_precision,
                            context_recall, faithfulness)

from src.config import DATA_DIR


def load_eval_set(path: Path) -> list[dict]:
    """
    Expected format, one entry per question:
    {
      "question": "...",
      "ground_truth": "...",
    }
    """
    return json.loads(path.read_text())


def run_pipeline_and_collect(pipeline, eval_set: list[dict]) -> dict:
    """Runs the full RAG pipeline on each eval question and packages
    results into the format RAGAS expects."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in eval_set:
        result = pipeline.answer(item["question"])
        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append([c.chunk.text for c in result.chunks])
        ground_truths.append(item["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def run_eval(pipeline, eval_set_path: Path, label: str = "run") -> dict:
    eval_set = load_eval_set(eval_set_path)
    data = run_pipeline_and_collect(pipeline, eval_set)
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    print(f"\n=== Eval results: {label} ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    out_path = DATA_DIR / f"eval_results_{label}.json"
    out_path.write_text(json.dumps(scores, indent=2))
    return scores


def compare_runs(baseline_scores: dict, improved_scores: dict) -> None:
    print("\n=== Before vs After ===")
    for metric in baseline_scores:
        before = baseline_scores[metric]
        after = improved_scores.get(metric, float("nan"))
        delta = after - before
        print(f"  {metric}: {before:.3f} -> {after:.3f}  ({delta:+.3f})")


if __name__ == "__main__":
    # Example usage — wire up your two pipeline configs (baseline vs full) in app.py
    # and call run_eval twice, then compare_runs(). See README for the interview
    # narrative this is meant to produce.
    print("See README.md 'The interview story' section for how to use this harness.")
