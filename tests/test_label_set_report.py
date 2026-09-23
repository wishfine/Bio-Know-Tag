import json
import sys
from pathlib import Path

from scripts.report_label_set_comparison import main


def test_report_renders_per_label_rates(tmp_path: Path, monkeypatch):
    run = tmp_path / "ds"
    run.mkdir()
    report = {
        "counts": {
            "units": 100000,
            "compared": 99998,
            "missing_prediction": 2,
            "eligible": 99998,
            "exact": 1,
            "contains_legacy": 1,
            "subset_of_legacy": 1,
            "mixed_add_remove": 1,
            "disjoint": 1,
            "retained_assignments": 4,
            "legacy_assignments": 7,
        },
        "input_sha256": {"units": "unit-hash", "labels": "label-hash", "predictions": "prediction-hash"},
    }
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    label = {
        "label_id": "L",
        "label_name": "测试Label",
        "legacy_questions": 5,
        "predicted_questions": 4,
        "retained": 3,
        "missed": 2,
        "new_vs_legacy": 1,
        "retention_rate": 0.6,
        "miss_rate": 0.4,
        "relationship_counts_among_legacy_questions": {
            key: 1 for key in ("exact", "contains_legacy", "subset_of_legacy", "mixed_add_remove", "disjoint")
        },
        "relationship_rates_among_legacy_questions": {
            key: 0.2 for key in ("exact", "contains_legacy", "subset_of_legacy", "mixed_add_remove", "disjoint")
        },
    }
    (run / "per_label.jsonl").write_text(json.dumps(label) + "\n", encoding="utf-8")
    output = tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", ["report_label_set_comparison.py", "--run", f"DS={run}", "--output", str(output)])
    assert main() == 0
    text = output.read_text(encoding="utf-8")
    assert "测试Label" in text
    assert "完全一致" in text
    assert "3 / 60.00%" in text
    assert "1 / 20.00%" in text
