#!/usr/bin/env python3
"""One vote per model, then both second votes only on cross-model disagreement."""
import argparse
import json
from pathlib import Path

from bio_know_tag.strict_adaptive_vote_runner import run_strict_adaptive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('units','candidates','run-dir'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--labels',type=Path,default=Path('configs/labels.jsonl'))
    parser.add_argument('--seed-from-six-vote',type=Path)
    parser.add_argument('--allow-legacy-missing-prompt-hash',action='store_true',help='Explicitly allow old evidence without prompt SHA; version/map/input validated but actual old prompt unverified')
    parser.add_argument('--qwen-endpoint',default='http://172.22.0.35:9204/v1/chat/completions')
    parser.add_argument('--qwen-model',default='qwen3.8-27b-fp8')
    parser.add_argument('--ds-endpoint',default='http://172.22.0.35:9205/v1/chat/completions')
    parser.add_argument('--ds-model',default='ds-v4-flash')
    parser.add_argument('--workers-per-vote',type=int,default=35)
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--qwen-timeout',type=float,default=600)
    parser.add_argument('--ds-timeout',type=float,default=300)
    parser.add_argument('--retries',type=int,default=5)
    parser.add_argument('--retry-delay',type=float,default=1)
    parser.add_argument('--third-diagnostic-limit',type=int,default=0,help='0 disables third votes; deterministic sample of <=10000 stage2 questions')
    parser.add_argument('--diagnostic-seed',default='strict-third-v1')
    parser.add_argument('--skip-model-preflight',action='store_true')
    args=vars(parser.parse_args())
    preflight=not args.pop('skip_model_preflight')
    units,candidates,labels,run_dir=(args.pop(x) for x in ('units','candidates','labels','run_dir'))
    result=run_strict_adaptive(units,candidates,labels,run_dir,preflight=preflight,**args)
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
