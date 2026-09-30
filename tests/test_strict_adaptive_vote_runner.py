import importlib
import json
from pathlib import Path

import pytest


def api():
    try:
        return importlib.import_module('bio_know_tag.strict_adaptive_vote_runner')
    except ModuleNotFoundError:
        pytest.fail('strict adaptive runner is not implemented')


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def pred(qid, ids, **flags):
    return dict(question_id=qid, selected_labels=[{'label_id': x} for x in ids], **flags)


def fixture_inputs(tmp):
    units, candidates = tmp/'u.jsonl', tmp/'c.jsonl'
    write(units, [{'question_id': q, 'stem': 'actual question', 'unit_type': 'standalone'} for q in ['a','b','c']])
    write(candidates, [{'question_id': q, 'candidates': [{'label_id':x} for x in ['A','B','C']]} for q in ['a','b','c']])
    write(tmp/'votes/qwen1/predictions.jsonl', [pred('a',['A','B']),pred('b',['A','B']),pred('c',[],context_insufficient=True)])
    write(tmp/'votes/ds1/predictions.jsonl', [pred('a',['B','A']),pred('b',['A','C']),pred('c',[])])
    return units, candidates


def test_route_only_cross_model_set_disagreement(tmp_path):
    u,c=fixture_inputs(tmp_path)
    r=api().build_stage2(u,c,tmp_path)
    assert r['input']==3 and r['disagreements']==1 and r['quality_review']==1
    assert [json.loads(x)['question_id'] for x in (tmp_path/'routing/stage2/units.jsonl').read_text().splitlines()]==['b']
    assert api().build_stage2(u,c,tmp_path)==r


def test_strict_intersection_and_quality_no_training_auto_approval(tmp_path):
    u,c=fixture_inputs(tmp_path)
    api().build_stage2(u,c,tmp_path)
    write(tmp_path/'votes/qwen2_disagreement/predictions.jsonl',[pred('b',['A','B'])])
    write(tmp_path/'votes/ds2_disagreement/predictions.jsonl',[pred('b',['A','B'])])
    r=api().aggregate_strict(u,c,tmp_path)
    rows=list(map(json.loads,(tmp_path/'consensus/predictions.jsonl').read_text().splitlines()))
    assert r['input']==3
    assert [x['label_id'] for x in rows[0]['selected_labels']]==['A','B']
    assert [x['label_id'] for x in rows[1]['selected_labels']]==['A']
    assert rows[1]['actual_vote_count']==4
    assert rows[1]['label_vote_counts']['B']=={'qwen':2,'ds':1}
    assert rows[2]['quality_status']=='REVIEW' and rows[2]['actual_vote_count']==2
    assert all(not x['usable_for_training'] for x in rows)


def test_third_diagnostic_only_shrinks_and_never_fabricates_votes(tmp_path):
    u,c=fixture_inputs(tmp_path);api().build_stage2(u,c,tmp_path)
    write(tmp_path/'votes/qwen2_disagreement/predictions.jsonl',[pred('b',['A','B'])])
    write(tmp_path/'votes/ds2_disagreement/predictions.jsonl',[pred('b',['A','B'])])
    r=api().build_stage3(tmp_path, limit=1, seed='test')
    assert r['selected']==1
    write(tmp_path/'votes/qwen3_diagnostic/predictions.jsonl',[pred('b',['B'])])
    write(tmp_path/'votes/ds3_diagnostic/predictions.jsonl',[pred('b',['A','B'])])
    api().aggregate_strict(u,c,tmp_path)
    rows=list(map(json.loads,(tmp_path/'consensus/predictions.jsonl').read_text().splitlines()))
    assert rows[0]['actual_vote_count']==2 and rows[0]['third_vote_status']=='NOT_REQUESTED'
    assert rows[1]['actual_vote_count']==6 and rows[1]['selected_labels']==[]
    assert rows[1]['withdrawn_after_third_label_ids']==['A']


def test_missing_second_vote_not_treated_as_no(tmp_path):
    u,c=fixture_inputs(tmp_path);api().build_stage2(u,c,tmp_path)
    write(tmp_path/'votes/qwen2_disagreement/predictions.jsonl',[])
    write(tmp_path/'votes/ds2_disagreement/predictions.jsonl',[pred('b',['A'])])
    with pytest.raises(ValueError,match='missing|misaligned'):
        api().aggregate_strict(u,c,tmp_path)


def test_single_pair_ignores_existing_conditional_votes(tmp_path):
    u,c=fixture_inputs(tmp_path)
    # Existing second votes must neither trigger more work nor affect Q1/D1.
    write(tmp_path/'votes/qwen2_disagreement/predictions.jsonl',[pred('b',[])])
    write(tmp_path/'votes/ds2_disagreement/predictions.jsonl',[pred('b',[])])
    report=api().aggregate_strict(u,c,tmp_path,single_pair_only=True)
    rows=list(map(json.loads,(tmp_path/'consensus/predictions.jsonl').read_text().splitlines()))
    assert [x['label_id'] for x in rows[1]['selected_labels']]==['A']
    assert rows[1]['cross_model_disagreement'] and not rows[1]['stage2_requested']
    assert all(row['actual_vote_count']==2 for row in rows)
    assert report['four_vote_used']==report['six_vote_used']==0


def test_frozen_subset_tamper_refused(tmp_path):
    u,c=fixture_inputs(tmp_path);api().build_stage2(u,c,tmp_path)
    (tmp_path/'routing/stage2/units.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='changed'):
        api().build_stage2(u,c,tmp_path)


def test_prediction_outside_candidate_set_refused(tmp_path):
    u,c=fixture_inputs(tmp_path)
    write(tmp_path/'votes/ds1/predictions.jsonl',[pred('a',['X']),pred('b',['A']),pred('c',[])])
    with pytest.raises(ValueError,match='candidate'):
        api().build_stage2(u,c,tmp_path)


def test_controller_transport_resume_and_seed_old_votes(tmp_path, monkeypatch):
    """Only HTTP/process scheduling is replaced; real streaming persistence used."""
    from bio_know_tag.full_adjudication import run_full_adjudication
    from bio_know_tag.ds import DSResponse
    import hashlib
    from bio_know_tag.adjudication import build_adjudication_prompt
    runner=api()
    u,c=fixture_inputs(tmp_path)
    labels=tmp_path/'labels.jsonl'
    write(labels,[dict(label_id=x,label_name=x,label_path=x,definition=x,core_concepts='',distinctions='') for x in ['A','B','C']])
    old=tmp_path/'old';old.mkdir();(old/'controller.lock').touch()
    label_cards={r['label_id']:r for r in map(json.loads,labels.read_text().splitlines())}
    endpoints={'qwen':'http://q','ds':'http://d'}
    services={m:dict(endpoint=e,model=m,timeout=600 if m=='qwen' else 300,max_tokens=1024,retries=5) for m,e in endpoints.items()}
    global_manifest=runner._child_manifest(u,c,labels,services['qwen'],1)
    for model in ['qwen','ds']:
        for number in [1,2,3]:
            directory=old/'votes'/f'{model}{number}'
            directory.mkdir(parents=True)
            child=runner._child_manifest(u,c,labels,services[model],1)
            (directory/'run_manifest.json').write_text(json.dumps(child))
            records=[]
            for unit,candidate in runner._aligned([u,c]):
                qid=unit['question_id'];prompt,mapping=build_adjudication_prompt(unit,candidate['candidates'],label_cards)
                selected=['A','B'] if qid=='a' or model=='qwen' else ['A','C'] if qid=='b' and number==1 else ['A','B'] if qid=='b' else []
                if qid=='c':selected=[]
                codes=[code for code,lid in mapping.items() if lid in selected]
                records.append(dict(question_id=qid,model=model,prompt_version=runner._prompt_version_for_unit(unit),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    candidate_code_map=mapping,error=None,finish_reason='stop',parsed_response=dict(selected=codes,
                    evidence={code:'actual question' for code in codes},context_insufficient=False,need_expand_recall=False,reason='test')))
                records[-1]['parsed_response']['none_of_candidates']=not codes
                records[-1]['parsed_response']['unknown_selected_codes_dropped']=[]
            write(directory/'evidence.jsonl',records)
    originals={p:p.read_bytes() for p in old.rglob('*') if p.is_file()}
    http_calls=[];process_calls=[]

    def run_local(specs, *, labels, **kwargs):
        for vote,units,candidates,directory,service in specs:
            process_calls.append((vote,[r['question_id'] for r in runner._rows(units)]))
            class Client:
                def chat(self,*args,**kwargs):
                    http_calls.append(vote)
                    return DSResponse(content=json.dumps(dict(selected=[],evidence={},context_insufficient=False,need_expand_recall=False,reason='empty')),endpoint=service['endpoint'],attempts=1,latency_seconds=0.01,finish_reason='stop')
            child=runner._child_manifest(units,candidates,labels,service,1)
            run_full_adjudication(units,candidates,labels,directory,Client(),model=service['model'],workers=1,
                max_tokens=1024,request_config=child['request_config'])
    monkeypatch.setattr(runner,'_run_vote_processes',run_local)
    out=tmp_path/'strict'
    options=dict(seed_from_six_vote=old,qwen_endpoint='http://q',ds_endpoint='http://d',qwen_model='qwen',ds_model='ds',preflight=False,third_diagnostic_limit=1)
    report=runner.run_strict_adaptive(u,c,labels,out,**options)
    assert report['stage2']['disagreements']==1
    assert report['logical_requests']==10
    assert http_calls==[] # All four subset votes reuse the original identities.
    assert process_calls==[('qwen1',['a','b','c']),('ds1',['a','b','c']),('qwen2_disagreement',['b']),('ds2_disagreement',['b']),('qwen3_diagnostic',['b']),('ds3_diagnostic',['b'])]
    assert all(p.read_bytes()==content for p,content in originals.items())
    runner.run_strict_adaptive(u,c,labels,out,**options)
    assert len(process_calls)==6
    runner.run_strict_adaptive(u,c,labels,out,**{**options,'workers_per_vote':64})
    assert len(process_calls)==6
    single_options={**options,'third_diagnostic_limit':0,'single_pair_only':True,'workers_per_vote':64}
    # Diagnostic configuration remains immutable; the usual migration uses 0.
    with pytest.raises(ValueError,match='manifest mismatch'):
        runner.run_strict_adaptive(u,c,labels,out,**single_options)
    with pytest.raises(ValueError,match='manifest mismatch'):
        runner.run_strict_adaptive(u,c,labels,out,**{**options,'max_tokens':512})


def test_seed_requires_stopped_source_and_rejects_candidate_map_mismatch(tmp_path):
    import fcntl
    runner=api()
    u,c=fixture_inputs(tmp_path);labels=tmp_path/'labels.jsonl'
    write(labels,[dict(label_id=x) for x in ['A','B','C']])
    service=dict(endpoint='http://q',model='qwen',timeout=600,max_tokens=1024,retries=5)
    manifest=runner._child_manifest(u,c,labels,service,1)
    old=tmp_path/'old';vote=old/'votes/qwen1';vote.mkdir(parents=True)
    (vote/'run_manifest.json').write_text(json.dumps(manifest))
    write(vote/'evidence.jsonl',[dict(question_id='a',error=None,model='qwen',prompt_version=runner.PROMPT_VERSION,candidate_code_map={},parsed_response={})])
    with (old/'controller.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='still running'):
            runner._seed_vote(old,'qwen1',tmp_path/'new/votes/qwen1',u,c,labels,service,1,manifest)
    with pytest.raises(ValueError,match='incompatible seeded evidence'):
        runner._seed_vote(old,'qwen1',tmp_path/'new/votes/qwen1',u,c,labels,service,1,manifest)


def test_empty_stage2_never_starts_conditional_votes(tmp_path,monkeypatch):
    runner=api();u,c=fixture_inputs(tmp_path);labels=tmp_path/'labels.jsonl';write(labels,[dict(label_id=x) for x in ['A','B','C']])
    out=tmp_path/'run';calls=[]
    def local(specs, *, labels, **kwargs):
        for vote,units,candidates,directory,service in specs:
            calls.append(vote);write(directory/'predictions.jsonl',[pred(q['question_id'],['A']) for q in runner._rows(units)])
            child=runner._child_manifest(units,candidates,labels,service,1)
            (directory/'run_manifest.json').write_text(json.dumps(child))
            (directory/'report.json').write_text(json.dumps(dict(success=3,error=0,model=service['model'])))
    monkeypatch.setattr(runner,'_run_vote_processes',local)
    report=runner.run_strict_adaptive(u,c,labels,out,preflight=False)
    assert calls==['qwen1','ds1'] and report['logical_requests']==6
    assert report['stage2']['disagreements']==0


def test_resume_adaptive_as_single_pair_keeps_first_votes(tmp_path,monkeypatch):
    runner=api();u,c=fixture_inputs(tmp_path);labels=tmp_path/'labels.jsonl'
    write(labels,[dict(label_id=x) for x in ['A','B','C']])
    out=tmp_path/'run';calls=[]
    def local(specs, *, labels, **kwargs):
        for vote,units,candidates,directory,service in specs:
            calls.append(vote)
            rows=list(runner._rows(units))
            selected=['A','B'] if vote.startswith('qwen') else ['A','C']
            write(directory/'predictions.jsonl',[pred(q['question_id'],selected) for q in rows])
            child=runner._child_manifest(units,candidates,labels,service,1)
            (directory/'run_manifest.json').write_text(json.dumps(child))
            (directory/'report.json').write_text(json.dumps(dict(success=len(rows),error=0,model=service['model'])))
    monkeypatch.setattr(runner,'_run_vote_processes',local)
    runner.run_strict_adaptive(u,c,labels,out,preflight=False)
    assert calls==['qwen1','ds1','qwen2_disagreement','ds2_disagreement']
    originals={v:(out/'votes'/v/'predictions.jsonl').read_bytes() for v in runner.FIRST}
    result=runner.run_strict_adaptive(u,c,labels,out,preflight=False,single_pair_only=True,workers_per_vote=64)
    assert len(calls)==4 and result['logical_requests']==6
    assert list(out.glob('run_manifest.previous-*.json'))
    assert all((out/'votes'/v/'predictions.jsonl').read_bytes()==data for v,data in originals.items())
    assert not any(row['stage2_requested'] for row in runner._rows(out/'consensus/predictions.jsonl'))


def test_quality_gate_retains_audit_but_blocks_candidate(tmp_path):
    u,c=fixture_inputs(tmp_path)
    rows=list(map(json.loads,u.read_text().splitlines()));rows[0]['flags']={'parent_context_missing':True};write(u,rows)
    api().build_stage2(u,c,tmp_path)
    write(tmp_path/'votes/qwen2_disagreement/predictions.jsonl',[pred('b',['A'])])
    write(tmp_path/'votes/ds2_disagreement/predictions.jsonl',[pred('b',['A'])])
    api().aggregate_strict(u,c,tmp_path)
    first=json.loads((tmp_path/'consensus/predictions.jsonl').read_text().splitlines()[0])
    assert len(first['selected_labels'])==2 and not first['high_precision_candidate']


def test_seed_legacy_hash_requires_explicit_opt_in_and_orphans_block(tmp_path,monkeypatch):
    runner=api();u,c=fixture_inputs(tmp_path);labels=tmp_path/'labels.jsonl'
    write(labels,[dict(label_id=x) for x in ['A','B','C']])
    service=dict(endpoint='http://q',model='qwen',timeout=600,max_tokens=1024,retries=5)
    manifest=runner._child_manifest(u,c,labels,service,1)
    old=tmp_path/'old';vote=old/'votes/qwen1';vote.mkdir(parents=True);(old/'controller.lock').touch()
    (vote/'run_manifest.json').write_text(json.dumps(manifest))
    label_cards={r['label_id']:r for r in runner._rows(labels)}
    unit,candidate=next(runner._aligned([u,c]))
    prompt,mapping=runner.build_adjudication_prompt(unit,candidate['candidates'],label_cards)
    record=dict(question_id='a',model='qwen',prompt_version=runner._prompt_version_for_unit(unit),candidate_code_map=mapping,error=None,finish_reason='stop',
        parsed_response=dict(selected=[],evidence={},none_of_candidates=True,context_insufficient=False,need_expand_recall=False,reason='none'))
    write(vote/'evidence.jsonl',[record]);target=tmp_path/'new/votes/qwen1'
    monkeypatch.setattr(runner,'_source_processes',lambda source:[123])
    with pytest.raises(RuntimeError,match='orphan'):
        runner._seed_vote(old,'qwen1',target,u,c,labels,service,1,manifest)
    monkeypatch.setattr(runner,'_source_processes',lambda source:[])
    with pytest.raises(ValueError,match='missing prompt hash'):
        runner._seed_vote(old,'qwen1',target,u,c,labels,service,1,manifest)
    assert not target.exists()
    runner._seed_vote(old,'qwen1',target,u,c,labels,service,1,manifest,allow_legacy_missing_prompt_hash=True)
    assert json.loads((target/'seed_provenance.json').read_text())['prompt_hash_unverified_rows']==1
    record['prompt_sha256']='wrong';write(vote/'evidence.jsonl',[record])
    with pytest.raises(ValueError,match='prompt hash mismatch'):
        runner._seed_vote(old,'qwen1',tmp_path/'other/votes/qwen1',u,c,labels,service,1,manifest,allow_legacy_missing_prompt_hash=True)


def test_seed_source_changed_while_copying_not_published(tmp_path,monkeypatch):
    runner=api();u,c=fixture_inputs(tmp_path);labels=tmp_path/'labels.jsonl'
    write(labels,[dict(label_id=x) for x in ['A','B','C']])
    service=dict(endpoint='http://q',model='qwen',timeout=600,max_tokens=1024,retries=5)
    manifest=runner._child_manifest(u,c,labels,service,1)
    old=tmp_path/'old';vote=old/'votes/qwen1';vote.mkdir(parents=True);(old/'controller.lock').touch()
    (vote/'run_manifest.json').write_text(json.dumps(manifest));write(vote/'evidence.jsonl',[])
    checks=[]
    def writers(source):
        checks.append(1)
        if len(checks)==2:
            (vote/'evidence.jsonl').write_text('{}\n')
        return []
    monkeypatch.setattr(runner,'_source_processes',writers)
    target=tmp_path/'new/votes/qwen1'
    with pytest.raises(RuntimeError,match='source changed'):
        runner._seed_vote(old,'qwen1',target,u,c,labels,service,1,manifest)
    assert not target.exists()
