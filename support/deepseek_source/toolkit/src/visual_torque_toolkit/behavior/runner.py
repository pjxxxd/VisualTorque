"""Interruptible behavior evaluation with append-only predictions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .adapter import GenerationResult, ModelAdapter
from .data import BehaviorExample, load_behavior_examples
from .metrics import analyze_behavior_run
from .parsing import parse_choice_label
from .schema import BehaviorRunConfig


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"predictions.jsonl has invalid JSON at line {line_number}"
                ) from error
            example_id = record["example_id"]
            if example_id in records:
                raise ValueError(f"duplicate prediction for {example_id}")
            records[example_id] = record
    return records


def _legacy_identity_matches(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    """Allow one-time migration from path-based Dataset identities.

    Old runs stored ``dataset_dir`` instead of a content fingerprint. Ignore
    only that legacy location field; every actual experiment setting must
    still match. The run config is rewritten with the new fingerprint before
    inference continues.
    """

    if "dataset_fingerprint" in previous:
        return False
    normalized_previous = dict(previous)
    normalized_previous.pop("dataset_dir", None)
    normalized_current = dict(current)
    normalized_current.pop("dataset_fingerprint", None)
    return normalized_previous == normalized_current


def _changed_identity_fields(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    return sorted(
        key
        for key in previous.keys() | current.keys()
        if previous.get(key) != current.get(key)
    )


def _chunks(values: list[BehaviorExample], size: int) -> Iterable[list[BehaviorExample]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _choice_probabilities(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {label: math.exp(score - maximum) for label, score in scores.items()}
    normalizer = sum(weights.values())
    return {label: value / normalizer for label, value in weights.items()}


def _prediction_record(
    example: BehaviorExample,
    scores: dict[str, float],
    scoring_mode: str,
    latency_seconds: float,
    generated_response: str | None,
    generated_raw_response: str | None,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    predicted_label = max(sorted(scores), key=lambda label: scores[label])
    semantics = example.label_to_semantic
    predicted_semantic = semantics[predicted_label]
    alternatives = [score for label, score in scores.items() if label != example.gold_label]
    gold_margin = scores[example.gold_label] - max(alternatives)

    semantic_to_label = {semantic: label for label, semantic in semantics.items()}
    signed_margin = None
    if "left" in semantic_to_label and "right" in semantic_to_label:
        signed_margin = (
            scores[semantic_to_label["left"]] - scores[semantic_to_label["right"]]
        )

    generated_label = parse_choice_label(generated_response, example.labels)
    generated_semantic = semantics.get(generated_label) if generated_label else None
    return {
        "schema_version": "behavior-1.0",
        "record_type": "behavior_prediction",
        "example_id": example.example_id,
        "model_id": model_metadata.get("model_id"),
        "model_revision": model_metadata.get("resolved_revision")
        or model_metadata.get("requested_revision"),
        "condition": example.condition.value,
        "question_id": example.question_id,
        "scene_id": example.scene_id,
        "question_kind": example.question_kind,
        "variant": example.variant,
        "split": example.split,
        "problem_type": example.problem_type,
        "gold_outcome": example.outcome,
        "net_torque": example.net_torque,
        "legal_labels": list(example.labels),
        "label_to_semantic": semantics,
        "gold_label": example.gold_label,
        "gold_semantic": example.gold_semantic,
        "candidate_logprobs": scores,
        "choice_probabilities": _choice_probabilities(scores),
        "scoring_mode": scoring_mode,
        "predicted_label": predicted_label,
        "predicted_semantic": predicted_semantic,
        "likelihood_correct": predicted_semantic == example.gold_semantic,
        "correct_vs_best_alternative_margin": gold_margin,
        "signed_left_right_margin": signed_margin,
        "generated_raw_response": generated_raw_response,
        "generated_response": generated_response,
        "generated_label": generated_label,
        "generated_semantic": generated_semantic,
        "generation_parse_success": (
            generated_label is not None if generated_response is not None else None
        ),
        "generation_correct": (
            generated_semantic == example.gold_semantic
            if generated_response is not None and generated_label is not None
            else None
        ),
        "latency_seconds": latency_seconds,
    }


def run_behavior_evaluation(
    config: BehaviorRunConfig,
    adapter: ModelAdapter,
) -> dict[str, Any]:
    """Run or resume an evaluation, then regenerate every analysis report."""

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    metadata_path = output_dir / "model_metadata.json"
    predictions_path = output_dir / "predictions.jsonl"

    identity = config.identity_record()
    if config_path.exists():
        previous = _read_json(config_path)
        previous_identity = previous.get("identity", {})
        if previous_identity != identity and not _legacy_identity_matches(
            previous_identity, identity
        ):
            changed = ", ".join(
                _changed_identity_fields(previous_identity, identity)
            )
            raise ValueError(
                "existing output directory belongs to a different experiment; "
                f"differing identity fields: {changed}. Choose a new output directory."
            )
        if not config.resume:
            raise FileExistsError(
                "run already exists and resume=False; choose a new output directory"
            )
    elif any(output_dir.iterdir()):
        raise FileExistsError(
            "refusing to mix a behavior run with an unrecognized non-empty directory"
        )

    examples = load_behavior_examples(config)
    model_metadata = adapter.metadata
    _write_json(
        config_path,
        {
            "schema_version": "behavior-1.0",
            "identity": identity,
            "selected_example_count": len(examples),
        },
    )
    if metadata_path.exists():
        previous_metadata = _read_json(metadata_path)
        stable_fields = (
            "model_id",
            "requested_revision",
            "resolved_revision",
            "dtype",
            "transformers_version",
        )
        changed = [
            field
            for field in stable_fields
            if previous_metadata.get(field) != model_metadata.get(field)
        ]
        if changed:
            raise ValueError(
                "model environment changed while resuming this run "
                f"({', '.join(changed)}); choose a new output directory"
            )
    else:
        _write_json(metadata_path, model_metadata)

    existing = _read_existing_predictions(predictions_path)
    expected_ids = {example.example_id for example in examples}
    unexpected = set(existing) - expected_ids
    if unexpected:
        raise ValueError(
            f"predictions file contains {len(unexpected)} examples outside this run"
        )
    pending = [example for example in examples if example.example_id not in existing]
    original_index = {example.example_id: index for index, example in enumerate(examples)}

    pending_by_condition: dict[str, list[BehaviorExample]] = {}
    for example in pending:
        pending_by_condition.setdefault(example.condition.value, []).append(example)

    batch_count = sum(
        math.ceil(len(group) / config.batch_size)
        for group in pending_by_condition.values()
    )
    batches: Iterable[list[BehaviorExample]] = (
            batch
            for condition in sorted(pending_by_condition)
            for batch in _chunks(pending_by_condition[condition], config.batch_size)
    )
    if batch_count:
        try:
            from tqdm.auto import tqdm

            batches = tqdm(
                batches,
                total=batch_count,
                desc=f"VisualTorque inference ({len(pending)} examples remaining)",
                unit="batch",
                dynamic_ncols=True,
            )
        except ImportError:
            pass

    with predictions_path.open("a", encoding="utf-8") as handle:
        for batch in batches:
            score_results = adapter.score_batch(batch)
            if len(score_results) != len(batch):
                raise RuntimeError("adapter returned the wrong number of score results")

            generation_examples = [
                example
                for example in batch
                if config.generation_every
                and original_index[example.example_id] % config.generation_every == 0
            ]
            generated = adapter.generate_batch(generation_examples)
            if len(generated) != len(generation_examples):
                raise RuntimeError("adapter returned the wrong number of generations")
            response_by_id: dict[str, tuple[str, str]] = {
                example.example_id: (
                    (response.raw_response, response.parsed_content)
                    if isinstance(response, GenerationResult)
                    else (str(response), str(response))
                )
                for example, response in zip(generation_examples, generated)
            }

            for example, result in zip(batch, score_results):
                if set(result.logprob_by_label) != set(example.labels):
                    raise RuntimeError(
                        f"candidate scores do not match labels for {example.example_id}"
                    )
                raw_and_parsed = response_by_id.get(example.example_id)
                record = _prediction_record(
                    example,
                    result.logprob_by_label,
                    result.scoring_mode,
                    result.latency_seconds,
                    raw_and_parsed[1] if raw_and_parsed else None,
                    raw_and_parsed[0] if raw_and_parsed else None,
                    model_metadata,
                )
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    progress = {
        "expected": len(examples),
        "completed": len(_read_existing_predictions(predictions_path)),
        "newly_completed": len(pending),
    }
    _write_json(output_dir / "progress.json", progress)
    reports = analyze_behavior_run(
        config.dataset_dir,
        predictions_path,
        output_dir,
    )
    return {"progress": progress, "reports": reports}
