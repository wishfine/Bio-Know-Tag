#!/usr/bin/env python3
"""Estimate completion time of all six full-file votes from live, read-only outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.full_vote_eta import estimate_full_votes


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "待估算"
    whole = max(0, round(seconds))
    days, remainder = divmod(whole, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{days}天{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="adjudication directory containing votes/")
    parser.add_argument("--total", type=int, required=True, help="expected questions per vote")
    parser.add_argument("--window", type=int, default=1000, help="recent evidence records per vote")
    parser.add_argument("--json", action="store_true", help="output machine-readable JSON")
    args = parser.parse_args()
    result = estimate_full_votes(args.run_dir, total=args.total, window=args.window)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("六票进度（最近速度是估算；成功数最多比实时落后 1,000 次请求）")
    print("票次    阶段                 成功/总数       进度     最近成功/分钟   预计剩余")
    for vote in result["votes"]:
        success = vote["success_at_last_report"]
        done = f"{success:,}/{args.total:,}" if success is not None else "尚无进度报告"
        percent = f"{vote['progress_percent']:.2f}%" if vote["progress_percent"] is not None else "-"
        rate = vote["recent_successes_per_second"]
        per_minute = f"{rate * 60:.1f}" if rate is not None else "-"
        print(
            f"{vote['vote']:<7} {vote['phase']:<20} {done:<16} "
            f"{percent:>7} {per_minute:>13}   {_duration(vote['estimated_remaining_seconds'])}"
        )
    if result["estimated_finish_at"]:
        finish = datetime.fromisoformat(result["estimated_finish_at"]).astimezone()
        print(f"整体由最慢的 {result['slowest_vote']} 决定；预计剩余 {_duration(result['estimated_remaining_seconds'])}")
        print(f"预计完成时间（服务器本地时区）：{finish:%Y-%m-%d %H:%M:%S %Z}")
    else:
        print("至少一票尚未形成足够的成功响应，暂不能给出可靠的整体完成时间。")
    quiet = [
        vote["vote"] for vote in result["votes"]
        if vote["phase"] != "complete"
        and vote["evidence_age_seconds"] is not None
        and vote["evidence_age_seconds"] > 600
    ]
    if quiet:
        print(f"注意：{', '.join(quiet)} 的证据文件超过 10 分钟未更新，当前 ETA 可能失准。")
    print("说明：取每票最近最多 1,000 条完整证据估速；重启、服务停顿或负载变化都会改变预测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
