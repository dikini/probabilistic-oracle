from dataclasses import asdict
import json
from pathlib import Path
import pytest
from probabilistic_oracle.bench import synthetic_queries, load_queries, run_benchmark, read_run
from probabilistic_oracle.oracle import ScoreResult

WORLD = {'id':'reference', 'split':'eval', 'counts':[16, 4, 8, 72]}


def test_finite_world_targets_and_conditioning():
    queries = synthetic_queries(WORLD)
    assert len(queries) == 18
    base = {q.kind:q for q in queries if q.variant == 'base'}
    assert {k:q.target for k,q in base.items()} == pytest.approx({
        'h':.2, 'not_h':.8, 'e_h':.8, 'e_not_h':.1, 'posterior':2/3, 'joint':.16})
    assert base['e_h'].request.assumptions and not base['e_h'].request.observations
    assert base['posterior'].request.observations and not base['posterior'].request.assumptions
    assert {q.case_id for q in queries} == {'reference'}
    assert {q.split for q in queries} == {'eval'}


@pytest.mark.parametrize('counts', [[0,0,5,5], [1,-1,2,3], [1,2,3], [1.5,2,3,4]])
def test_bad_worlds_rejected(counts):
    with pytest.raises(ValueError):
        synthetic_queries(dict(WORLD, counts=counts))


def test_duplicate_case_cannot_leak_across_splits(tmp_path):
    p=tmp_path/'cases.json'
    p.write_text(json.dumps([WORLD, dict(WORLD, split='calibration')]))
    with pytest.raises(ValueError, match='Duplicate'):
        load_queries(p)


class FixtureOracle:
    provenance = {'backend': 'test-fixture'}
    def score(self, request):
        return ScoreResult(request, 'fixture prompt', [1], [2,3], [-.69314718056]*2, .5, 1., [2], .01, self.provenance)


def test_run_roundtrip_and_explicit_failures(tmp_path):
    class FailingOracle(FixtureOracle):
        def score(self, request):
            if request.assumptions:
                raise ValueError('deliberate fixture failure')
            return super().score(request)
    queries=synthetic_queries(WORLD)[:6]
    run=tmp_path/'run'
    run_benchmark(queries, FailingOracle(), run)
    manifest, records=read_run(run)
    assert manifest['backend']['backend']=='test-fixture'
    assert len(records)==len(queries)
    assert sum(r['status']=='error' for r in records)==2
    assert all('error' in r for r in records if r['status']=='error')
    assert manifest['queries']==[json.loads(json.dumps(asdict(q))) for q in queries]
    with pytest.raises(FileExistsError):
        run_benchmark(queries, FixtureOracle(), run)


def test_truncated_run_is_detected(tmp_path):
    run=tmp_path/'run'
    run_benchmark(synthetic_queries(WORLD)[:2], FixtureOracle(), run)
    p=run/'records.jsonl'
    p.write_text(p.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='Incomplete'):
        read_run(run)


def test_manifest_target_changes_are_detected(tmp_path):
    run=tmp_path/'run'
    run_benchmark(synthetic_queries(WORLD),FixtureOracle(),run)
    path=run/'manifest.json'
    data=json.loads(path.read_text())
    data['queries'][0]['target']=.9
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash'):
        read_run(run)
