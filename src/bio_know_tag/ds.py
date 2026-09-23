"""DeepSeek/OpenAI-compatible client and label-alignment schemas."""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ALIGNMENT_DECISIONS = {
    "原释义更准确",
    "DS释义更准确",
    "两者基本等价",
    "两者都有问题",
    "无法仅凭现有信息判断",
}
NAME_SUFFICIENCY_DECISIONS = {
    "名称本身足够",
    "需要原释义",
    "标签体系有问题",
}
TAXONOMY_ISSUE_DECISIONS = {
    "DS释义更准确",
    "两者都有问题",
    "无法仅凭现有信息判断",
}


@dataclass(frozen=True)
class DSResponse:
    content: str
    endpoint: str
    attempts: int
    latency_seconds: float
    usage: dict[str, Any] | None = None
    reasoning: Any = None
    response_message_keys: tuple[str, ...] = ()
    retry_errors: tuple[dict[str, Any], ...] = ()


class DSRequestError(RuntimeError):
    """Terminal request failure with diagnostics from every HTTP attempt."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        endpoint: str,
        latency_seconds: float,
        retry_errors: Iterable[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.endpoint = endpoint
        self.latency_seconds = latency_seconds
        self.retry_errors = tuple(retry_errors)


class DSClient:
    """Small retrying client for an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        endpoints: Iterable[str],
        model: str,
        *,
        timeout: float = 120,
        retries: int = 3,
        retry_delay: float = 0.25,
        request_interval: float = 0.0,
        enable_thinking: bool | None = None,
        max_in_flight_per_endpoint: int | None = None,
    ) -> None:
        self.endpoints = [endpoint.rstrip("/") for endpoint in endpoints if endpoint]
        if not self.endpoints:
            raise ValueError("at least one endpoint is required")
        if retries < 1:
            raise ValueError("retries must be at least 1")
        if request_interval < 0:
            raise ValueError("request_interval must be non-negative")
        if max_in_flight_per_endpoint is not None and max_in_flight_per_endpoint < 1:
            raise ValueError("max_in_flight_per_endpoint must be positive")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.request_interval = request_interval
        self.enable_thinking = enable_thinking
        self.max_in_flight_per_endpoint = max_in_flight_per_endpoint
        self._endpoint_slots = {
            endpoint: threading.BoundedSemaphore(max_in_flight_per_endpoint)
            for endpoint in set(self.endpoints)
        } if max_in_flight_per_endpoint is not None else {}
        self._next_endpoint = 0
        self._endpoint_lock = threading.Lock()
        self._request_slot_lock = threading.Lock()
        self._next_request_time = 0.0

    def _wait_for_request_slot(self) -> None:
        if not self.request_interval:
            return
        with self._request_slot_lock:
            now = time.monotonic()
            request_time = max(now, self._next_request_time)
            self._next_request_time = request_time + self.request_interval
        delay = request_time - now
        if delay > 0:
            time.sleep(delay)

    def _defer_retry_slot(self) -> None:
        """Keep a full interval after a failed request before retrying."""
        if not self.request_interval:
            return
        with self._request_slot_lock:
            self._next_request_time = max(
                self._next_request_time,
                time.monotonic() + self.request_interval,
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
    ) -> DSResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.monotonic()
        last_error: Exception | None = None
        retry_errors: list[dict[str, Any]] = []
        with self._endpoint_lock:
            starting_index = self._next_endpoint
            self._next_endpoint = (self._next_endpoint + 1) % len(self.endpoints)

        for attempt in range(1, self.retries + 1):
            endpoint_index = (starting_index + attempt - 1) % len(self.endpoints)
            endpoint = self.endpoints[endpoint_index]
            request = Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                self._wait_for_request_slot()
                slot = self._endpoint_slots.get(endpoint)
                with slot if slot is not None else nullcontext():
                    with urlopen(request, timeout=self.timeout) as response:
                        response_body = json.loads(response.read().decode("utf-8"))
                message = response_body["choices"][0]["message"]
                content = message["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("empty chat completion content")
                return DSResponse(
                    content=content,
                    endpoint=endpoint,
                    attempts=attempt,
                    latency_seconds=round(time.monotonic() - started, 3),
                    usage=response_body.get("usage"),
                    reasoning=message.get("reasoning", message.get("reasoning_content")),
                    response_message_keys=tuple(message),
                    retry_errors=tuple(retry_errors),
                )
            except (HTTPError, URLError, TimeoutError, OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                retry_errors.append(
                    {
                        "attempt": attempt,
                        "endpoint": endpoint,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if attempt < self.retries:
                    self._defer_retry_slot()
                if attempt < self.retries and self.retry_delay:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(delay * random.uniform(0.8, 1.2))

        endpoint = retry_errors[-1]["endpoint"] if retry_errors else self.endpoints[0]
        raise DSRequestError(
            f"chat completion failed after {self.retries} attempts: {last_error}",
            attempts=self.retries,
            endpoint=endpoint,
            latency_seconds=round(time.monotonic() - started, 3),
            retry_errors=retry_errors,
        ) from last_error


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse one JSON object from plain text or a Markdown code fence."""
    text = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    object_start = text.find("{")
    if object_start < 0:
        raise ValueError("response does not contain a JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[object_start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _validate_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return value


def validate_stage1_result(value: dict[str, Any]) -> dict[str, Any]:
    for field in ("core_meaning", "included_content", "excluded_content"):
        if field not in value:
            raise ValueError(f"missing {field}")
    _validate_string(value["core_meaning"], "core_meaning")
    _validate_string_list(value["included_content"], "included_content")
    _validate_string_list(value["excluded_content"], "excluded_content")
    return value


def validate_alignment_result(value: dict[str, Any]) -> dict[str, Any]:
    required = (
        "alignment_score",
        "omissions",
        "expansions",
        "boundary_differences",
        "audit_decision",
        "audit_reason",
        "name_sufficiency",
    )
    for field in required:
        if field not in value:
            raise ValueError(f"missing {field}")
    score = value["alignment_score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("alignment_score must be an integer between 1 and 5")
    for field in ("omissions", "expansions", "boundary_differences"):
        _validate_string_list(value[field], field)
    if value["audit_decision"] not in ALIGNMENT_DECISIONS:
        raise ValueError("audit_decision is not one of the documented decisions")
    if value["name_sufficiency"] not in NAME_SUFFICIENCY_DECISIONS:
        raise ValueError(
            "name_sufficiency is not one of the documented decisions"
        )
    _validate_string(value["audit_reason"], "audit_reason")
    return value


def classify_alignment(value: dict[str, Any]) -> str:
    validated = validate_alignment_result(value)
    if (
        validated["alignment_score"] <= 3
        or validated["audit_decision"] in TAXONOMY_ISSUE_DECISIONS
        or validated["name_sufficiency"] == "标签体系有问题"
    ):
        return "L3"
    if validated["name_sufficiency"] == "需要原释义":
        return "L2"
    return "L1"


def build_stage1_prompt(label_name: str) -> str:
    return f"""你是一名高中生物教师。

现在给你一个高中生物知识点标签：
【{label_name}】

在不知道任何已有知识点释义的情况下，仅根据标签名称和你的高中生物知识，写出你认为这个标签对应的知识范围。

请只输出一个 JSON 对象，字段严格如下：
{{
  "core_meaning": "核心含义",
  "included_content": ["应该包含的考查内容"],
  "excluded_content": ["不应该包含的相近内容"]
}}

不要猜测标签体系设计者的特殊规则。不要输出 Markdown 或 JSON 之外的文字。"""


def build_alignment_prompt(label: dict[str, str], generated: dict[str, Any]) -> str:
    original = {
        "definition": label["definition"],
        "core_concepts": label["core_concepts"],
        "common_assessments": label["common_assessments"],
        "distinctions": label["distinctions"],
    }
    return f"""你是一名严谨的高中生物知识点体系审核专家。

请比较同一 Label 的老师原释义与“仅看 Label 名”生成的 DS 释义。不要默认任一方必然正确。

Label 名称：{label['label_name']}
老师原释义：{json.dumps(original, ensure_ascii=False)}
DS 生成释义：{json.dumps(generated, ensure_ascii=False)}

对齐分标准：
5 基本完全一致；4 核心一致，仅边界有少量差异；3 主体一致但有重要缺失或扩张；2 理解方向明显偏差；1 基本不是同一知识点。

逐项核验规则：
1. 声称“遗漏”前，必须检查 DS 的 core_meaning、included_content、excluded_content 三处；已经明确或同义表达的内容不得算遗漏。
2. 声称“扩张”前，必须检查老师原释义的 definition、core_concepts、common_assessments、distinctions 四处；合理举例、同义改写和解释细化不得算重要扩张。
3. 轻微措辞差异、示例多少和解释详略不等于“需要原释义”。只有该差异可能改变一道题该不该打此 Label 时，才判“需要原释义”。
4. name_sufficiency 必须回答实际判标问题：只给名称能否稳定决定题目是否属于该 Label。

请只输出一个 JSON 对象：
{{
  "alignment_score": 1到5的整数,
  "omissions": ["DS 遗漏内容；没有则空数组"],
  "expansions": ["DS 多理解内容；没有则空数组"],
  "boundary_differences": ["关键边界差异；没有则空数组"],
  "audit_decision": "原释义更准确|DS释义更准确|两者基本等价|两者都有问题|无法仅凭现有信息判断",
  "audit_reason": "简洁说明依据",
  "name_sufficiency": "名称本身足够|需要原释义|标签体系有问题"
}}

不要输出 Markdown 或 JSON 之外的文字。"""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(value)
    return records


def load_completed_ids(path: str | Path, id_field: str = "label_id") -> set[str]:
    evidence = Path(path)
    if not evidence.exists():
        return set()
    completed: set[str] = set()
    for record in read_jsonl(evidence):
        if not record.get("error") and isinstance(record.get("parsed_response"), dict):
            completed.add(str(record[id_field]))
    return completed


def append_evidence(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def summarize_evidence(
    path: str | Path,
    expected_ids: Iterable[str],
    *,
    id_field: str = "label_id",
) -> dict[str, int]:
    expected = [str(value) for value in expected_ids]
    expected_set = set(expected)
    latest: dict[str, dict[str, Any]] = {}
    evidence_rows = 0
    evidence = Path(path)
    if evidence.exists():
        for record in read_jsonl(evidence):
            evidence_rows += 1
            record_id = str(record.get(id_field, ""))
            if record_id in expected_set:
                latest[record_id] = record
    success = sum(
        not record.get("error") and isinstance(record.get("parsed_response"), dict)
        for record in latest.values()
    )
    errors = sum(bool(record.get("error")) for record in latest.values())
    processed = len(latest)
    return {
        "input": len(expected),
        "processed": processed,
        "success": success,
        "error": errors,
        "pending": len(expected) - processed,
        "evidence_rows": evidence_rows,
    }
