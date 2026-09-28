#!/usr/bin/env python3
"""Stop only Python vote processes whose --run-dir belongs to one exact run."""

import argparse
import os
import signal
import time
from pathlib import Path


def matching_processes(root: Path) -> dict[int, list[str]]:
    matches = {}
    allowed = {
        "run_qwen_ds_six_votes_full.py", "run_candidate_adjudication_full.py",
        "run_qwen_ds_adaptive_votes_full.py",
        "run_qwen_ds_strict_adaptive_full.py",
    }
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [value.decode() for value in (entry / "cmdline").read_bytes().split(b"\0") if value]
            if not any(Path(value).name in allowed for value in argv):
                continue
            value = next((arg.split("=", 1)[1] for arg in argv if arg.startswith("--run-dir=")), None)
            if value is None:
                if "--run-dir" not in argv:
                    continue
                value = argv[argv.index("--run-dir") + 1]
            directory = Path(value)
            if not directory.is_absolute():
                directory = (entry / "cwd").resolve() / directory
            directory = directory.resolve()
            if directory == root or root in directory.parents:
                matches[int(entry.name)] = argv
        except (OSError, ValueError, IndexError, UnicodeDecodeError):
            continue
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stop", action="store_true", help="Send SIGTERM; otherwise only print exact matches")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if not (root / "run_manifest.json").is_file():
        parser.error("exact run directory must contain run_manifest.json")
    processes = matching_processes(root)
    for pid, argv in processes.items():
        print(f"PID={pid} {' '.join(argv)}", flush=True)
    if not args.stop:
        return 0
    for pid in processes:
        # Revalidate before signaling; never kill unrelated model servers.
        if pid in matching_processes(root):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        remaining = matching_processes(root)
        if not remaining:
            print("All matching vote processes stopped; result files preserved.", flush=True)
            return 0
        time.sleep(0.5)
    print(f"Still running: {sorted(matching_processes(root))}; do not start migration yet.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
