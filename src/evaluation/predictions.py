"""Durable per-question predictions and model-independent evaluation reports."""

from collections import Counter
import json
import math
import os
from pathlib import Path
import torch

from .metrics import MetricTotals, question_log_probs, question_metrics


def prediction(context_id, question, output):
    return dict(context_id=context_id, question_id=question.id,
                option_ids=[o.id for o in question.options],
                targets=question.target.probabilities, category=question.category,
                task=str(question.task), target_kind=str(question.target.kind), **output)


QUESTION_FIELDS = (
    "context_id", "question_id", "option_ids", "targets", "category", "task", "target_kind",
)


def question_metadata(row):
    """Keep only alignment metadata; never retain scores or reasoning in the index."""
    return {key: row[key] for key in QUESTION_FIELDS}


def read_predictions(path):
    """Stream one prediction at a time, including potentially large reasoning."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def load_reference(path):
    if path is None:
        return None
    rows = {}
    for row in read_predictions(path):
        identity = row["question_id"]
        if identity in rows:
            raise ValueError("Duplicate reference question")
        rows[identity] = question_metadata(row)
    if not rows:
        raise ValueError("Reference predictions are empty")
    return rows


def check_question(expected, existing):
    """Reject changed question metadata when comparing runs or continuing a journal."""
    if existing is None:
        return
    keys = tuple(key for key in QUESTION_FIELDS if key != "targets")
    if any(expected[key] != existing[key] for key in keys):
        raise ValueError(f"Question metadata differs: {expected['question_id']}")
    targets, saved = torch.tensor(expected["targets"]), torch.tensor(existing["targets"])
    if targets.shape != saved.shape or not torch.allclose(targets, saved, atol=1e-7, rtol=1e-6):
        raise ValueError(f"Question targets differ: {expected['question_id']}")


class PredictionJournal:
    """Durable JSONL on disk; only completed IDs and alignment metadata stay in RAM."""

    def __init__(self, path):
        self.path = Path(path)
        self.completed = {}
        if self.path.exists():
            # Interrupted writes can leave an incomplete final line. Only that
            # uncommitted tail is discarded; corruption in committed rows fails.
            with self.path.open("rb+") as stream:
                end = 0
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        stream.truncate(end)
                        break
                    row = json.loads(line)
                    if row["question_id"] in self.completed:
                        raise ValueError("Duplicate question in prediction journal")
                    self.completed[row["question_id"]] = question_metadata(row)
                    end = stream.tell()

    def iter_rows(self):
        """Rebuild metrics on resume without loading the complete prediction file."""
        if self.path.exists():
            yield from read_predictions(self.path)

    def append(self, row):
        if row["question_id"] in self.completed:
            raise ValueError("Question already recorded")
        scores, targets = row.get("scores"), row["targets"]
        if not targets or len(row["option_ids"]) != len(targets):
            raise ValueError("Prediction option/target alignment is invalid")
        if row["status"] == "ok" and (scores is None or len(scores) != len(targets)
                or not all(math.isfinite(x) for x in scores)):
            raise ValueError("Successful prediction requires finite scores per option")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.completed[row["question_id"]] = question_metadata(row)


class PredictionSummary:
    """Accumulate each prediction once; requesting a report does not replay rows."""

    def __init__(self, rows=()):
        self.totals = MetricTotals()
        self.counts, self.successes, self.correct, self.random = Counter(), Counter(), Counter(), Counter()
        self.histogram, self.statuses = Counter(), Counter()
        self.seconds = self.reasoning = 0
        self.profile = Counter()
        for row in rows:
            self.add(row)

    def add(self, row):
        k = len(row["targets"])
        self.histogram[k] += 1
        self.statuses[row["status"]] += 1
        self.seconds += row.get("seconds", 0)
        self.reasoning += row.get("reasoning_tokens", 0)
        # Old resumed rows must not inflate rates with tokens lacking timings.
        if "generation_seconds" in row and "seconds" in row:
            self.profile["profiled_questions"] += 1
            for key in ("generation_seconds", "prefill_seconds", "decode_seconds",
                        "scoring_seconds", "decode_tokens", "scoring_forward_tokens",
                        "scoring_forwards", "cache_reused_tokens", "cache_branch_copies"):
                self.profile[key] += row.get(key, 0)
            self.profile["generation_tokens"] += row.get("reasoning_tokens", 0)
            for key in ("peak_allocated_gib", "reserved_gib"):
                self.profile[key] = max(self.profile[key], row.get(key, 0))
        groups = ("all", f"category/{row['category']}", f"target/{row['target_kind']}")
        for group in groups:
            self.counts[group] += 1
            self.random[group] += 1 / k
        if row["status"] != "ok":
            return
        scores = torch.tensor(row["scores"], dtype=torch.float32)
        targets = torch.tensor(row["targets"], dtype=torch.float32)
        links = torch.zeros(k, dtype=torch.long)
        log_probs = question_log_probs(scores, links, 1)
        values = question_metrics(log_probs, targets, links, 1)
        self.totals.add(values, [row["category"]], [row["target_kind"]])
        for group in groups:
            self.successes[group] += 1
            self.correct[group] += float(values[0, 1])

    def result(self):
        """Accuracy includes failures; CE/Brier cover completed answers only."""
        metrics = self.totals.result()
        for group, count in self.counts.items():
            if group in self.successes:
                metrics[f"{group}/completed_accuracy"] = metrics[f"{group}/accuracy"]
            metrics[f"{group}/accuracy"] = self.correct[group] / count
            metrics[f"{group}/coverage"] = self.successes[group] / count
            metrics[f"{group}/random_accuracy"] = self.random[group] / count
        metrics.update({"evaluation/questions": self.counts["all"],
                        "evaluation/seconds": self.seconds,
                        "evaluation/reasoning_tokens": self.reasoning})
        if self.seconds > 0:
            metrics["evaluation/questions_per_second"] = self.counts["all"] / self.seconds
            metrics["evaluation/reasoning_tokens_per_second"] = self.reasoning / self.seconds
        if self.profile:
            metrics.update({f"evaluation/{key}": value for key, value in self.profile.items()})
            for phase in ("generation", "decode"):
                seconds = self.profile[f"{phase}_seconds"]
                if seconds > 0:
                    metrics[f"evaluation/{phase}_tokens_per_second"] = self.profile[f"{phase}_tokens"] / seconds
        return dict(metrics=metrics, option_counts=dict(sorted(self.histogram.items())),
                    statuses=dict(self.statuses),
                    metric_scope="Accuracy includes failures as incorrect; CE/Brier use successful questions only.")
