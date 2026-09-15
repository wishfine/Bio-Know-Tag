import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bio_know_tag.ds import (
    ALIGNMENT_DECISIONS,
    NAME_SUFFICIENCY_DECISIONS,
    DSClient,
    build_stage1_prompt,
    classify_alignment,
    load_completed_ids,
    parse_json_content,
    validate_alignment_result,
    validate_stage1_result,
)


def test_parse_json_content_accepts_markdown_fence():
    result = parse_json_content('```json\n{"core_meaning":"复制"}\n```')

    assert result["core_meaning"] == "复制"


def test_validate_stage1_result_requires_all_fields():
    with pytest.raises(ValueError, match="excluded_content"):
        validate_stage1_result(
            {"core_meaning": "复制", "included_content": ["复制过程"]}
        )


def test_build_stage1_prompt_only_contains_label_name():
    prompt = build_stage1_prompt("DNA半保留复制")

    assert "DNA半保留复制" in prompt
    assert "原释义" not in prompt
    assert "核心含义" in prompt


def test_completed_ids_only_includes_successful_rows(tmp_path: Path):
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        "\n".join(
            (
                json.dumps({"label_id": "1", "parsed_response": {"ok": True}, "error": None}),
                json.dumps({"label_id": "2", "parsed_response": None, "error": "timeout"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_completed_ids(evidence) == {"1"}


def test_ds_client_retries_retryable_http_error():
    state = {"requests": 0, "payload": None}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            state["requests"] += 1
            length = int(self.headers["Content-Length"])
            state["payload"] = json.loads(self.rfile.read(length))
            if state["requests"] == 1:
                self.send_response(500)
                self.end_headers()
                return
            content = json.dumps(
                {
                    "core_meaning": "DNA复制方式",
                    "included_content": ["亲代链保留"],
                    "excluded_content": ["全保留复制"],
                },
                ensure_ascii=False,
            )
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    try:
        client = DSClient([endpoint], "DeepSeek-V4-Flash", timeout=2, retries=2)
        response = client.chat([{"role": "user", "content": "test"}], max_tokens=64)
    finally:
        server.shutdown()
        thread.join()

    assert response.attempts == 2
    assert response.endpoint == endpoint
    assert state["requests"] == 2
    assert state["payload"]["temperature"] == 0
    assert state["payload"]["model"] == "DeepSeek-V4-Flash"


def test_ds_client_distributes_concurrent_requests_across_endpoints():
    counts = [0, 0]
    locks = [threading.Lock(), threading.Lock()]
    servers = []
    threads = []

    def handler_for(index):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers["Content-Length"])
                self.rfile.read(length)
                with locks[index]:
                    counts[index] += 1
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": '{"ok":true}'}}],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):  # noqa: A002
                return

        return Handler

    try:
        for index in range(2):
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(index))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
        endpoints = [
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            for server in servers
        ]
        client = DSClient(endpoints, "model", timeout=2, retries=1)
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(
                executor.map(
                    lambda _: client.chat([{"role": "user", "content": "x"}]),
                    range(4),
                )
            )
    finally:
        for server in servers:
            server.shutdown()
        for thread in threads:
            thread.join()

    assert counts == [2, 2]
    assert all(response.usage["total_tokens"] == 5 for response in responses)


def test_validate_alignment_rejects_out_of_range_score():
    result = {
        "alignment_score": 6,
        "omissions": [],
        "expansions": [],
        "boundary_differences": [],
        "audit_decision": "原释义更准确",
        "audit_reason": "边界更完整",
        "name_sufficiency": "需要原释义",
    }

    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_alignment_result(result)


def test_validate_alignment_rejects_unknown_decision():
    result = {
        "alignment_score": 3,
        "omissions": ["边界"],
        "expansions": [],
        "boundary_differences": [],
        "audit_decision": "都可以",
        "audit_reason": "",
        "name_sufficiency": "需要原释义",
    }

    with pytest.raises(ValueError, match="audit_decision"):
        validate_alignment_result(result)


@pytest.mark.parametrize("decision", sorted(ALIGNMENT_DECISIONS))
def test_validate_alignment_accepts_documented_decisions(decision: str):
    result = {
        "alignment_score": 4,
        "omissions": [],
        "expansions": [],
        "boundary_differences": [],
        "audit_decision": decision,
        "audit_reason": "依据",
        "name_sufficiency": "名称本身足够",
    }

    assert validate_alignment_result(result) == result


def test_classify_alignment_separates_clear_boundary_and_taxonomy_issue():
    base = {
        "alignment_score": 5,
        "omissions": [],
        "expansions": [],
        "boundary_differences": [],
        "audit_decision": "两者基本等价",
        "audit_reason": "一致",
        "name_sufficiency": "名称本身足够",
    }

    assert classify_alignment(base) == "L1"
    assert classify_alignment({**base, "boundary_differences": ["非关键措辞差异"]}) == "L1"
    assert classify_alignment({**base, "name_sufficiency": "需要原释义"}) == "L2"
    assert classify_alignment({**base, "alignment_score": 3}) == "L3"
    assert classify_alignment({**base, "audit_decision": "DS释义更准确"}) == "L3"
    assert classify_alignment({**base, "name_sufficiency": "标签体系有问题"}) == "L3"


def test_validate_alignment_rejects_unknown_name_sufficiency():
    result = {
        "alignment_score": 4,
        "omissions": [],
        "expansions": [],
        "boundary_differences": [],
        "audit_decision": "两者基本等价",
        "audit_reason": "核心一致",
        "name_sufficiency": "大概可以",
    }

    with pytest.raises(ValueError, match="name_sufficiency"):
        validate_alignment_result(result)


@pytest.mark.parametrize("decision", sorted(NAME_SUFFICIENCY_DECISIONS))
def test_validate_alignment_accepts_name_sufficiency_decisions(decision: str):
    result = {
        "alignment_score": 4,
        "omissions": [],
        "expansions": [],
        "boundary_differences": [],
        "audit_decision": "两者基本等价",
        "audit_reason": "核心一致",
        "name_sufficiency": decision,
    }

    assert validate_alignment_result(result) == result
