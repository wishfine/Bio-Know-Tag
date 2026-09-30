"""Bounded-memory Q1/D1 -> conditional Q2/D2, strict per-Label intersection.

Uses the existing streaming single-vote transport; never mutates old vote runs.
All phases finish before freezing the next phase, avoiding live-file races.
"""
from __future__ import annotations

import fcntl
import hashlib
import heapq
import itertools
import json
import os
import signal
import sqlite3
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from bio_know_tag.adjudication import (
    PROMPT_VERSION, PARENT_PROMPT_VERSION, build_adjudication_prompt,
    _prompt_version_for_unit, validate_adjudication_result,
)
from bio_know_tag.full_adaptive_vote_runner import _rows, _count_rows, _run_vote_processes, _vote_complete
from bio_know_tag.six_vote_runner import _interrupt_on_sigterm, _write_json_atomic
from bio_know_tag.three_vote_runner import _model_preflight, _sha256

VERSION = 'adaptive-q1d1-strict-intersection-v1'
FIRST = ('qwen1', 'ds1')
SECOND = ('qwen2_disagreement', 'ds2_disagreement')
THIRD = ('qwen3_diagnostic', 'ds3_diagnostic')


def _aligned(paths):
    """Fail closed for partial, reordered or differently sized predictions."""
    for number, rows in enumerate(itertools.zip_longest(*(_rows(Path(p)) for p in paths)), 1):
        if any(r is None for r in rows):
            raise ValueError(f'missing or misaligned rows at {number}')
        qid = str(rows[0].get('question_id') or '')
        if not qid or any(str(r.get('question_id') or '') != qid for r in rows):
            raise ValueError(f'missing or misaligned question_id at {number}')
        yield rows


def _ids(row, candidate):
    cards = row.get('selected_labels')
    if not isinstance(cards, list) or row.get('error'):
        raise ValueError(f'invalid prediction: {row.get("question_id")}')
    ids = [str(c.get('label_id') or '') for c in cards]
    available = {str(c['label_id']) for c in candidate['candidates']}
    if any(not x for x in ids) or len(set(ids)) != len(ids) or not set(ids) <= available:
        raise ValueError(f'invalid or outside candidate label: {row.get("question_id")}')
    return set(ids)


def _quality(unit, votes):
    reasons = set()
    flags = unit.get('flags') or {}
    for key in ('parent_context_missing', 'content_review_required', 'needs_content_review', 'text_ineligible'):
        if flags.get(key) or unit.get(key):
            reasons.add(key)
    fields = ('stem', 'options', 'analysis', 'answer_text') if unit.get('unit_type') == 'composite_parent_extra' else ('stem',)
    if not any(str(unit.get(f) or '').strip() for f in fields):
        reasons.add('actual_question_text_missing')
    for vote in votes:
        for key in ('context_insufficient', 'need_expand_recall', 'needs_review', 'text_content_missing'):
            if vote.get(key):
                reasons.add(key)
        if vote.get('unknown_selected_codes_dropped'):
            reasons.add('unknown_selected_codes_dropped')
    return sorted(reasons)


def _write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def _frozen_report(directory, source_hashes):
    path = directory/'report.json'
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report['source_sha256'] != source_hashes:
        raise ValueError('routing source changed after freezing')
    for name, digest in report['output_sha256'].items():
        target = directory/name
        if not target.is_file() or _sha256(target) != digest:
            raise ValueError(f'frozen routing output changed: {target}')
    return report


def _publish_route(directory, counts, source_hashes, names):
    for name in names:
        (directory/f'.{name}.tmp').replace(directory/name)
    report = dict(counts, strategy_version=VERSION, source_sha256=source_hashes,
                  output_sha256={name:_sha256(directory/name) for name in names})
    _write_json_atomic(directory/'report.json', report)
    return report


def build_stage2(units, candidates, output):
    output = Path(output)
    paths = [Path(units), Path(candidates), *(output/'votes'/v/'predictions.jsonl' for v in FIRST)]
    hashes = {str(i):_sha256(p) for i,p in enumerate(paths)}
    directory = output/'routing/stage2'
    previous = _frozen_report(directory, hashes)
    if previous is not None:
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    names = ['units.jsonl','candidates.jsonl','routes.jsonl']
    counts = Counter(input=0, disagreements=0, quality_review=0, agreed_nonempty=0, both_empty=0)
    with ExitStack() as stack:
        handles = {n:stack.enter_context((directory/f'.{n}.tmp').open('w', encoding='utf-8')) for n in names}
        for unit,candidate,q,d in _aligned(paths):
            qs,ds = _ids(q,candidate),_ids(d,candidate)
            reasons = _quality(unit,[q,d])
            differs = qs != ds
            counts['input'] += 1
            counts['quality_review'] += bool(reasons)
            if differs:
                counts['disagreements'] += 1
                _write_row(handles['units.jsonl'],unit)
                _write_row(handles['candidates.jsonl'],candidate)
            else:
                counts['agreed_nonempty' if qs else 'both_empty'] += 1
            _write_row(handles['routes.jsonl'], dict(question_id=unit['question_id'],
                       stage2_requested=differs, quality_reasons=reasons))
            if counts['input'] % 250000 == 0:
                print(f"stage2 routing: {counts['input']:,}; disagreements={counts['disagreements']:,}", flush=True)
    return _publish_route(directory,counts,hashes,names)


def build_stage3(output, *, limit=0, seed='strict-third-v1'):
    """Deterministic hash-priority sample from stage2; never all by default."""
    output = Path(output)
    if not 0 <= limit <= 10000:
        raise ValueError('third diagnostic limit must be 0..10000')
    directory = output/'routing/stage3'
    stage2 = output/'routing/stage2'
    paths = [stage2/'units.jsonl',stage2/'candidates.jsonl']
    hashes = {str(i):_sha256(p) for i,p in enumerate(paths)}
    hashes.update(limit=limit, seed=seed)
    previous = _frozen_report(directory, hashes)
    if previous is not None:
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    heap = []
    if limit:
        for unit,candidate in _aligned(paths):
            qid = str(unit['question_id'])
            key = int(hashlib.sha256(f'{seed}\0{qid}'.encode()).hexdigest(),16)
            item = (-key,qid)
            if len(heap) < limit:
                heapq.heappush(heap,item)
            elif item > heap[0]:
                heapq.heapreplace(heap,item)
    selected = {qid for _,qid in heap}
    names=['units.jsonl','candidates.jsonl']
    with ExitStack() as stack:
        handles={n:stack.enter_context((directory/f'.{n}.tmp').open('w',encoding='utf-8')) for n in names}
        for unit,candidate in _aligned(paths):
            if str(unit['question_id']) in selected:
                _write_row(handles['units.jsonl'],unit)
                _write_row(handles['candidates.jsonl'],candidate)
    return _publish_route(directory,dict(selected=len(selected)),hashes,names)


def _take_pair(iterators, qid, candidate):
    rows = [next(it,None) for it in iterators]
    if any(r is None or str(r.get('question_id') or '') != qid for r in rows):
        raise ValueError(f'missing or misaligned conditional prediction: {qid}')
    for row in rows:
        _ids(row,candidate)
    return rows


def aggregate_strict(units,candidates,output, *, single_pair_only=False):
    output = Path(output)
    second = [_rows(output/'votes'/v/'predictions.jsonl') if not single_pair_only and (output/'votes'/v/'predictions.jsonl').exists() else iter(()) for v in SECOND]
    third = [_rows(output/'votes'/v/'predictions.jsonl') if not single_pair_only and (output/'votes'/v/'predictions.jsonl').exists() else iter(()) for v in THIRD]
    third_units = _rows(output/'routing/stage3/units.jsonl') if not single_pair_only and (output/'routing/stage3/units.jsonl').exists() else iter(())
    next_third = next(third_units,None)
    paths = [Path(units),Path(candidates),*(output/'votes'/v/'predictions.jsonl' for v in FIRST)]
    directory = output/'consensus'
    directory.mkdir(exist_ok=True)
    temp = directory/'.predictions.jsonl.tmp'
    counts = Counter(input=0, first_pair_pass=0, four_vote_used=0, six_vote_used=0, selected_assignments=0, quality_review=0, empty=0)
    with temp.open('w',encoding='utf-8') as handle:
        for unit,candidate,q,d in _aligned(paths):
            qid = str(unit['question_id'])
            votes=[q,d]; vote_names=list(FIRST)
            differs = _ids(q,candidate) != _ids(d,candidate)
            if differs and not single_pair_only:
                votes += _take_pair(second,qid,candidate); vote_names += list(SECOND)
                counts['four_vote_used'] += 1
            before_third = set.intersection(*(_ids(r,candidate) for r in votes))
            third_requested = next_third is not None and str(next_third['question_id']) == qid
            if third_requested:
                if not differs:
                    raise ValueError('third vote must be a stage2 diagnostic')
                votes += _take_pair(third,qid,candidate); vote_names += list(THIRD)
                next_third = next(third_units,None)
                counts['six_vote_used'] += 1
            kept = set.intersection(*(_ids(r,candidate) for r in votes))
            reasons = _quality(unit,votes)
            cards = {str(c['label_id']):c for r in votes for c in r['selected_labels']}
            support = {}
            for name,row in zip(vote_names,votes):
                for lid in _ids(row,candidate):
                    support.setdefault(lid,dict(qwen=0,ds=0))[ 'qwen' if name.startswith('qwen') else 'ds'] += 1
            selected = [{**cards[lid], 'qwen_support':support[lid]['qwen'], 'ds_support':support[lid]['ds']} for lid in sorted(kept)]
            n=len(votes)
            counts['input'] += 1
            counts['first_pair_pass'] += not differs and bool(kept) and not reasons
            counts['selected_assignments'] += len(selected)
            counts['quality_review'] += bool(reasons)
            counts['empty'] += not bool(kept)
            _write_row(handle, dict(question_id=qid, parent_id=unit.get('parent_id',qid),
                unit_type=unit.get('unit_type'), selected_labels=selected,
                policy_version='dual-model-single-pair-intersection-v1' if single_pair_only else VERSION, actual_vote_count=n, completed_stage_vote_count=n,
                executed_vote_ids=vote_names, label_vote_counts=support,
                not_retained_label_ids=sorted(set(support)-kept),
                withdrawn_after_third_label_ids=sorted(before_third-kept) if third_requested else [],
                selection_status=f'STRICT_INTERSECTION_{n}' if kept else 'NO_POSITIVE_LABEL',
                stage2_requested=differs and not single_pair_only, cross_model_disagreement=differs, third_vote_status='COMPLETED' if third_requested else 'NOT_REQUESTED',
                quality_status='REVIEW' if reasons else 'PASS', quality_reasons=reasons,
                high_precision_candidate=bool(kept and not reasons), needs_review=bool(reasons),
                calibration_status='NO_TEACHER_GOLD', training_eligible=False, usable_for_training=False))
    if next_third is not None or any(next(it,None) is not None for it in [*second,*third,third_units]):
        raise ValueError('extra or misaligned conditional predictions')
    temp.replace(directory/'predictions.jsonl')
    report=dict(counts,policy_version='dual-model-single-pair-intersection-v1' if single_pair_only else VERSION,training_approved=0)
    _write_json_atomic(directory/'report.json',report)
    return report


def _child_manifest(units,candidates,labels,service,retry_delay):
    return dict(runner_version='full-adjudication-bounded-v1',model=service['model'],
        max_tokens=service['max_tokens'],temperature=0,stream=True,audited_exclusions=None,
        request_config=dict(endpoint=service['endpoint'],timeout=service['timeout'],retries=service['retries'],
                            retry_delay=retry_delay,request_interval=0,enable_thinking=False),
        prompt_versions_by_unit_type=dict(composite_parent_extra=PARENT_PROMPT_VERSION,other=PROMPT_VERSION),
        input_paths=dict(units=str(units),candidates=str(candidates),labels=str(labels)),
        input_sha256=dict(units=_sha256(units),candidates=_sha256(candidates),labels=_sha256(labels)))


def _source_processes(source):
    """Linux read-only /proc guard: controller lock alone misses orphan children."""
    proc=Path('/proc')
    if not proc.is_dir():
        return []
    found=[]
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name)==os.getpid():
            continue
        try:
            argv=[s.decode() for s in (item/'cmdline').read_bytes().split(b'\0') if s]
            if not any(Path(s).name in {'run_candidate_adjudication_full.py','run_qwen_ds_six_votes_full.py','run_qwen_ds_adaptive_votes_full.py','run_qwen_ds_strict_adaptive_full.py'} for s in argv):
                continue
            value=next((s.split('=',1)[1] for s in argv if s.startswith('--run-dir=')),None)
            if value is None:
                value=argv[argv.index('--run-dir')+1]
            directory=Path(value)
            if not directory.is_absolute():
                directory=(item/'cwd').resolve()/directory
            directory=directory.resolve()
            if directory==source or source in directory.parents:
                found.append(int(item.name))
        except (OSError,UnicodeDecodeError,ValueError,IndexError):
            continue
    return found


def _seed_vote(source, source_vote, target, units,candidates,labels,service,retry_delay,global_manifest, *, allow_legacy_missing_prompt_hash=False):
    """Filter old evidence to this phase on disk, then atomically publish a copy.

    Reuses old vote identity, not arbitrary successful attempts from other votes.
    Never copies SQLite or mutates source. Partial old votes remain resumable.
    """
    old = source/'votes'/source_vote
    old_manifest_path = old/'run_manifest.json'
    child = json.loads(old_manifest_path.read_text())
    expected = _child_manifest(units,candidates,labels,service,retry_delay)
    for key in ('model','max_tokens','temperature','stream','request_config','prompt_versions_by_unit_type','audited_exclusions'):
        if child.get(key) != expected[key]:
            raise ValueError(f'old vote configuration mismatch: {source_vote}/{key}')
    if child.get('input_sha256') != global_manifest['input_sha256'] or child.get('input_paths') != global_manifest['input_paths']:
        raise ValueError('old vote full input mismatch')
    identity=dict(source_vote=source_vote,source_manifest_sha256=_sha256(old_manifest_path),
                  target_input_sha256=expected['input_sha256'],source_run=str(source),
                  allow_legacy_missing_prompt_hash=allow_legacy_missing_prompt_hash)
    provenance=target/'seed_provenance.json'
    if target.exists():
        if not provenance.is_file() or json.loads(provenance.read_text())['identity'] != identity:
            raise ValueError('seed target exists without compatible provenance')
        return
    with (source/'controller.lock').open('r+') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('old six-vote controller still running; stop before seeding') from exc
        if _source_processes(source):
            raise RuntimeError('old vote processes still running, including orphan children; stop before seeding')
        source_evidence=old/'evidence.jsonl'
        source_before=source_evidence.stat()
        print(f'validate/filter old {source_vote} evidence for {target.name}',flush=True)
        target.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.strict-seed-',dir=target.parent))
        dbpath=staging/'filter.sqlite3'
        db=sqlite3.connect(dbpath)
        try:
            db.execute('CREATE TABLE wanted (qid TEXT PRIMARY KEY, code_map TEXT, version TEXT, prompt_hash TEXT)')
            label_cards={str(r['label_id']):r for r in _rows(labels)}
            for number,(unit,candidate) in enumerate(_aligned([units,candidates]),1):
                prompt,mapping=build_adjudication_prompt(unit,candidate['candidates'],label_cards)
                db.execute('INSERT INTO wanted VALUES (?,?,?,?)',(str(unit['question_id']),json.dumps(mapping),_prompt_version_for_unit(unit),hashlib.sha256(prompt.encode('utf-8')).hexdigest()))
                if number%250000==0:
                    print(f'{target.name} seed index: {number:,}',flush=True)
            db.commit();copied=0;unverified=0
            with (old/'evidence.jsonl').open('rb') as src,(staging/'evidence.jsonl').open('wb') as dst:
                for line in src:
                    if not line.endswith(b'\n'):
                        break
                    record=json.loads(line)
                    if record.get('error') or not isinstance(record.get('parsed_response'),dict):
                        continue
                    wanted=db.execute('SELECT code_map,version,prompt_hash FROM wanted WHERE qid=?',(str(record.get('question_id') or ''),)).fetchone()
                    if wanted is None:
                        continue
                    mapping=json.loads(wanted[0])
                    if record.get('model') != service['model'] or record.get('prompt_version') != wanted[1] or record.get('candidate_code_map') != mapping or record.get('finish_reason') == 'length':
                        raise ValueError(f'incompatible seeded evidence: {record.get("question_id")}')
                    if record.get('prompt_sha256'):
                        if record['prompt_sha256']!=wanted[2]:
                            raise ValueError(f'seed actual prompt hash mismatch: {record.get("question_id")}')
                    elif not allow_legacy_missing_prompt_hash:
                        raise ValueError('old evidence missing prompt hash; explicitly opt into --allow-legacy-missing-prompt-hash after verifying frozen prompt')
                    else:
                        unverified+=1
                    validate_adjudication_result(record['parsed_response'],set(mapping))
                    dst.write(line);copied+=1
            _write_json_atomic(staging/'run_manifest.json',expected)
            _write_json_atomic(staging/'seed_provenance.json',dict(identity=identity,copied_evidence_rows=copied,
                               prompt_hash_unverified_rows=unverified,
                               copied_sha256=_sha256(staging/'evidence.jsonl')))
        finally:
            db.close()
        dbpath.unlink()
        writers=_source_processes(source)
        source_after=source_evidence.stat()
        if writers or (source_before.st_ino,source_before.st_size,source_before.st_mtime_ns)!=(source_after.st_ino,source_after.st_size,source_after.st_mtime_ns):
            raise RuntimeError('old seed source changed while copying; stop all source writers')
        staging.replace(target)
        print(f'seed {source_vote} -> {target.name}: {copied:,} successful evidence rows copied',flush=True)


def run_strict_adaptive(units_path,candidates_path,labels_path,run_dir, *,
    seed_from_six_vote=None, allow_legacy_missing_prompt_hash=False, qwen_endpoint='http://172.22.0.35:9204/v1/chat/completions',
    qwen_model='qwen3.8-27b-fp8',ds_endpoint='http://172.22.0.35:9205/v1/chat/completions',
    ds_model='ds-v4-flash',workers_per_vote=35,max_tokens=1024,qwen_timeout=600,ds_timeout=300,
    retries=5,retry_delay=1,third_diagnostic_limit=0,diagnostic_seed='strict-third-v1',preflight=True,
    single_pair_only=False):
    if min(workers_per_vote,max_tokens,retries,qwen_timeout,ds_timeout)<=0 or retry_delay<0:
        raise ValueError('workers/tokens/retries/timeouts must be positive')
    if not 0 <= third_diagnostic_limit <= 10000:
        raise ValueError('third diagnostic limit must be 0..10000')
    if single_pair_only and third_diagnostic_limit:
        raise ValueError('single-pair-only cannot request third votes')
    if qwen_endpoint==ds_endpoint:
        raise ValueError('model endpoints must differ')
    units,candidates,labels,output=map(lambda p:Path(p).resolve(),[units_path,candidates_path,labels_path,run_dir])
    source=Path(seed_from_six_vote).resolve() if seed_from_six_vote else None
    if source and (source==output or source in output.parents or output in source.parents):
        raise ValueError('new run must be separate from seed source')
    for path in [units,candidates,labels]:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'missing/empty input: {path}')
    services={m:dict(endpoint=e,model=n,timeout=t,max_tokens=max_tokens,retries=retries) for m,e,n,t in [
        ('qwen',qwen_endpoint,qwen_model,qwen_timeout),('ds',ds_endpoint,ds_model,ds_timeout)]}
    manifest=dict(runner_version=VERSION,input_paths=dict(units=str(units),candidates=str(candidates),labels=str(labels)),
        input_sha256=dict(units=_sha256(units),candidates=_sha256(candidates),labels=_sha256(labels)),
        services=services,workers_per_vote=workers_per_vote,retry_delay=retry_delay,
        temperature=0,stream=True,thinking_override=dict(qwen=False,ds=False),
        prompt_versions_by_unit_type=dict(composite_parent_extra=PARENT_PROMPT_VERSION,other=PROMPT_VERSION),
        third_diagnostic_limit=third_diagnostic_limit,diagnostic_seed=diagnostic_seed,
        seed_from_six_vote=str(source) if source else None,audited_exclusions_applied=False)
    manifest['allow_legacy_missing_prompt_hash']=allow_legacy_missing_prompt_hash
    manifest['adjudication_code_sha256']=_sha256(Path(__file__).with_name('adjudication.py'))
    manifest['full_adjudication_code_sha256']=_sha256(Path(__file__).with_name('full_adjudication.py'))
    if single_pair_only:
        manifest['single_pair_only']=True
    output.mkdir(parents=True,exist_ok=True)
    old_signal=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,_interrupt_on_sigterm)
    try:
        with (output/'controller.lock').open('a+') as lock:
            try:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('another strict adaptive controller is running') from exc
            path=output/'run_manifest.json'
            if path.exists():
                previous=json.loads(path.read_text())
                if previous!=manifest:
                    # Concurrency and routing policy may change without changing
                    # any first-pair prompts, transport settings or input data.
                    comparable=lambda m:{k:v for k,v in m.items() if k not in ('workers_per_vote','single_pair_only')}
                    if comparable(previous)!=comparable(manifest):
                        raise ValueError('strict adaptive manifest mismatch on resume')
                    digest=hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest()[:16]
                    _write_json_atomic(output/f'run_manifest.previous-{digest}.json',previous)
                    _write_json_atomic(path,manifest)
            else:
                _write_json_atomic(path,manifest)
            if preflight:
                for service in services.values():
                    _model_preflight(service['endpoint'],service['model'])

            def run_phase(votes,u,c,count):
                specs=[]
                for vote in votes:
                    model='qwen' if vote.startswith('qwen') else 'ds'
                    service=services[model];destination=output/'votes'/vote
                    hashes=dict(units=_sha256(u),candidates=_sha256(c),labels=manifest['input_sha256']['labels'])
                    if source and count:
                        old_vote=f'{model}{1 if vote in FIRST else 2 if vote in SECOND else 3}'
                        _seed_vote(source,old_vote,destination,u,c,labels,service,retry_delay,manifest,
                                   allow_legacy_missing_prompt_hash=allow_legacy_missing_prompt_hash)
                    if not count:
                        print(f'skip {vote}: empty routed input',flush=True);continue
                    if _vote_complete(destination,count,service['model'],hashes,service,retry_delay):
                        print(f'skip completed {vote}',flush=True);continue
                    specs.append((vote,u,c,destination,service))
                _run_vote_processes(specs,labels=labels,output=output,workers=workers_per_vote,retry_delay=retry_delay)
                for vote in votes:
                    model='qwen' if vote.startswith('qwen') else 'ds'
                    if count and not _vote_complete(output/'votes'/vote,count,services[model]['model'],hashes,services[model],retry_delay):
                        raise RuntimeError(f'incomplete phase vote: {vote}; resume unchanged command')

            count=_count_rows(units)
            _write_json_atomic(output/'progress.json',dict(status='running',phase='first_pair',input=count))
            run_phase(FIRST,units,candidates,count)
            if single_pair_only:
                _write_json_atomic(output/'progress.json',dict(status='running',phase='single_pair_consensus',input=count))
                consensus=aggregate_strict(units,candidates,output,single_pair_only=True)
                result=dict(status='complete',policy_version='dual-model-single-pair-intersection-v1',
                            input=count,consensus=consensus,logical_requests=2*count,
                            max_in_flight_per_service=workers_per_vote,training_approved=0)
                _write_json_atomic(output/'report.json',result)
                _write_json_atomic(output/'progress.json',dict(status='complete',phase='complete',input=count))
                return result
            split=build_stage2(units,candidates,output)
            _write_json_atomic(output/'progress.json',dict(status='running',phase='second_pair',**split))
            directory=output/'routing/stage2'
            run_phase(SECOND,directory/'units.jsonl',directory/'candidates.jsonl',split['disagreements'])
            diagnostic=build_stage3(output,limit=third_diagnostic_limit,seed=diagnostic_seed)
            _write_json_atomic(output/'progress.json',dict(status='running',phase='third_diagnostic',**diagnostic))
            directory=output/'routing/stage3'
            run_phase(THIRD,directory/'units.jsonl',directory/'candidates.jsonl',diagnostic['selected'])
            consensus=aggregate_strict(units,candidates,output)
            unverified=sum(json.loads(p.read_text()).get('prompt_hash_unverified_rows',0)
                           for p in (output/'votes').glob('*/seed_provenance.json'))
            result=dict(status='complete',policy_version=VERSION,input=count,stage2=split,stage3=diagnostic,
                        consensus=consensus,logical_requests=2*count+2*split['disagreements']+2*diagnostic['selected'],
                        max_in_flight_per_service=workers_per_vote,training_approved=0,
                        seed_prompt_hash_unverified_rows=unverified)
            _write_json_atomic(output/'report.json',result)
            _write_json_atomic(output/'progress.json',dict(status='complete',phase='complete',input=count))
            return result
    finally:
        signal.signal(signal.SIGTERM,old_signal)
