import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bio_know_tag.six_vote_runner import run_six_vote_shards


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "sharded"
    shard = root / "shards" / "00001"
    shard.mkdir(parents=True)
    (shard / "units.jsonl").write_text('{"question_id":"q1"}\n', encoding="utf-8")
    (shard / "candidates.jsonl").write_text('{"question_id":"q1","candidates":[]}\n', encoding="utf-8")
    (root / "report.json").write_text(json.dumps({
        "input": 1, "shards": [{"shard_id": "00001", "count": 1}],
    }), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"label_id":"L1"}\n', encoding="utf-8")
    return root, labels


def test_starts_all_six_votes_before_waiting_with_per_vote_limit(tmp_path: Path, monkeypatch):
    root, labels = _inputs(tmp_path)
    commands = []
    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
        def wait(self):
            assert len(commands) == 6
            return 0
    monkeypatch.setattr("bio_know_tag.six_vote_runner.subprocess.Popen", Process)
    result = run_six_vote_shards(root, labels, preflight=False)

    assert result["shards_completed_this_run"] == 1
    assert result["max_client_in_flight_per_service"] == 75
    assert result["max_client_in_flight_total"] == 150
    dirs = {command[command.index("--run-dir") + 1].split("/")[-1] for command in commands}
    assert dirs == {"qwen1", "qwen2", "qwen3", "ds1", "ds2", "ds3"}
    for command in commands:
        assert command[command.index("--workers") + 1] == "25"
        assert "--stream" in command
        assert "--no-audited-exclusions" in command
        assert "--disable-thinking" not in command
    assert len({command[command.index("--units") + 1] for command in commands}) == 1
    assert len({command[command.index("--candidates") + 1] for command in commands}) == 1
    qwen = [command for command in commands if command[command.index("--run-dir") + 1].endswith("qwen1")][0]
    ds = [command for command in commands if command[command.index("--run-dir") + 1].endswith("ds1")][0]
    assert qwen[qwen.index("--endpoint") + 1].endswith(":9204/v1/chat/completions")
    assert qwen[qwen.index("--model") + 1] == "qwen3.8-27b-fp8"
    assert qwen[qwen.index("--max-tokens") + 1] == "1024"
    assert ds[ds.index("--endpoint") + 1].endswith(":9205/v1/chat/completions")
    assert ds[ds.index("--model") + 1] == "ds-v4-flash"
    assert ds[ds.index("--max-tokens") + 1] == "512"


def test_resume_rejects_changed_model_without_launching(tmp_path: Path, monkeypatch):
    root, labels = _inputs(tmp_path)
    monkeypatch.setattr("bio_know_tag.six_vote_runner.subprocess.Popen", lambda *a, **k: type(
        "Process", (), {"wait": lambda self: 0}
    )())
    run_six_vote_shards(root, labels, preflight=False)
    with pytest.raises(ValueError, match="manifest mismatch"):
        run_six_vote_shards(root, labels, preflight=False, ds_model="different")


def test_one_failed_vote_stops_before_next_shard(tmp_path: Path, monkeypatch):
    root, labels = _inputs(tmp_path)
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            self.vote = Path(command[command.index("--run-dir") + 1]).name
        def wait(self, timeout=None):
            return 1 if self.vote == "ds2" else 0
        def poll(self):
            return 0
    monkeypatch.setattr("bio_know_tag.six_vote_runner.subprocess.Popen", Process)
    with pytest.raises(RuntimeError, match="ds2"):
        run_six_vote_shards(root, labels, preflight=False)
    assert len(calls) == 6


def test_interruption_terminates_all_live_votes(tmp_path: Path, monkeypatch):
    root, labels = _inputs(tmp_path)
    processes = []
    class Process:
        def __init__(self, command, **kwargs):
            self.terminated = False
            processes.append(self)
        def wait(self, timeout=None):
            if timeout is None and not self.terminated:
                raise KeyboardInterrupt
            return 0
        def poll(self):
            return 0 if self.terminated else None
        def terminate(self):
            self.terminated = True
    monkeypatch.setattr("bio_know_tag.six_vote_runner.subprocess.Popen", Process)
    with pytest.raises(KeyboardInterrupt):
        run_six_vote_shards(root, labels, preflight=False)
    assert len(processes) == 6
    assert all(process.terminated for process in processes)


def test_sigterm_handler_reaps_children_and_restores_handler(tmp_path: Path, monkeypatch):
    root, labels = _inputs(tmp_path)
    previous = signal.getsignal(signal.SIGTERM)
    processes = []
    class Process:
        def __init__(self, command, **kwargs):
            self.terminated = False
            processes.append(self)
        def wait(self, timeout=None):
            if timeout is None and not self.terminated:
                handler = signal.getsignal(signal.SIGTERM)
                assert callable(handler)
                handler(signal.SIGTERM, None)
            return 0
        def poll(self):
            return 0 if self.terminated else None
        def terminate(self):
            self.terminated = True
    monkeypatch.setattr("bio_know_tag.six_vote_runner.subprocess.Popen", Process)
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        run_six_vote_shards(root, labels, preflight=False)
    assert all(process.terminated for process in processes)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_real_sse_requests_use_two_services_and_resume_without_resending(tmp_path: Path):
    root, labels = _inputs(tmp_path)
    requests = {"qwen-test": [], "ds-test": []}
    servers = []
    threads = []

    def start_server(model: str) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": [{"id": model}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests[model].append(payload)
                answer = json.dumps({
                    "selected": [], "evidence": {}, "context_insufficient": False,
                    "need_expand_recall": False, "reason": "没有直接匹配的候选",
                }, ensure_ascii=False)
                parts = [
                    {"choices": [{"delta": {"content": answer}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
                body = ("".join("data: " + json.dumps(part, ensure_ascii=False) + "\n\n" for part in parts)
                        + "data: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
        return f"http://127.0.0.1:{server.server_port}/v1/chat/completions"

    qwen_endpoint = start_server("qwen-test")
    ds_endpoint = start_server("ds-test")
    try:
        kwargs = dict(
            qwen_endpoint=qwen_endpoint, qwen_model="qwen-test",
            ds_endpoint=ds_endpoint, ds_model="ds-test",
            workers_per_vote=1, qwen_max_tokens=64, ds_max_tokens=64,
            qwen_retries=1, ds_retries=1,
        )
        result = run_six_vote_shards(root, labels, qwen_timeout=5, ds_timeout=5, **kwargs)
        assert result["shards_completed_this_run"] == 1
        assert len(requests["qwen-test"]) == len(requests["ds-test"]) == 3
        for payload in requests["qwen-test"] + requests["ds-test"]:
            assert payload["stream"] is True
            assert payload["temperature"] == 0
        for vote in ("qwen1", "qwen2", "qwen3", "ds1", "ds2", "ds3"):
            vote_dir = root / "shards" / "00001" / "votes" / vote
            report = json.loads((vote_dir / "report.json").read_text(encoding="utf-8"))
            assert report["success"] == report["input"] == 1
            assert report["error"] == 0
        evidence = root / "shards" / "00001" / "votes" / "qwen1" / "evidence.jsonl"
        with evidence.open("ab") as handle:
            handle.write(b'{"interrupted":')
        run_six_vote_shards(root, labels, qwen_timeout=5, ds_timeout=5, **kwargs)
        assert len(requests["qwen-test"]) == len(requests["ds-test"]) == 3
        assert evidence.read_bytes().endswith(b"\n")
        tails = list(evidence.parent.glob("evidence.incomplete-tail.*.bin"))
        assert len(tails) == 1
        assert tails[0].read_bytes() == b'{"interrupted":'
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
