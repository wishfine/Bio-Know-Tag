import importlib.util
import sys
from pathlib import Path


def test_full_vote_cli_explicitly_disables_thinking(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_candidate_adjudication_full.py"
    spec = importlib.util.spec_from_file_location("full_adjudication_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    class Client:
        def __init__(self, endpoints, model, **kwargs):
            captured["client_kwargs"] = kwargs

    def run(*args, **kwargs):
        captured["request_config"] = kwargs["request_config"]
        return {"error": 0, "success": 1}

    monkeypatch.setattr(module, "DSClient", Client)
    monkeypatch.setattr(module, "run_full_adjudication", run)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--units", "units.jsonl", "--candidates", "candidates.jsonl",
        "--run-dir", "vote", "--endpoint", "http://fake/v1/chat/completions",
        "--model", "model-test", "--disable-thinking",
    ])
    assert module.main() == 0
    assert captured["client_kwargs"]["enable_thinking"] is False
    assert captured["request_config"]["enable_thinking"] is False
