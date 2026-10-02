import json
from pathlib import Path

from speedrun_dlm._hellaswag import reference_mask_counts
from speedrun_dlm.prepare_hellaswag import preprocess


def test_hellaswag_preprocessing_matches_harness_examples():
    assert preprocess("  walking [title] [noise]  home  ") == "walking.  home"


def test_reference_mask_schedule_covers_valid_counts():
    values = reference_mask_counts(target_tokens=7, batch_size=16, start=3)
    assert len(values) == 16
    assert values.min() >= 1
    assert values.max() <= 7


def test_rule_has_one_hellaswag_finish_clause():
    root = Path(__file__).resolve().parents[1]
    rule = json.loads((root / "finish_line_rule.json").read_text())
    clause_keys = {key for key in rule if key.endswith("_min")}
    assert clause_keys == {"hellaswag_accuracy_min"}
    assert rule["hellaswag_accuracy_min"] == 0.275


def test_rule_uses_ten_runs_without_fixed_seed_identifiers():
    root = Path(__file__).resolve().parents[1]
    rule = json.loads((root / "finish_line_rule.json").read_text())
    assert rule["training_runs_per_recipe"] == 10
    assert not any("seed" in key for key in rule)
