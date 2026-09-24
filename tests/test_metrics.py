import math
import pytest
from probabilistic_oracle.metrics import quality, calibrate, fit_temperature, interval, summarize
from probabilistic_oracle.bench import synthetic_queries, run_benchmark
from probabilistic_oracle.oracle import ScoreResult


def test_quality_against_hand_calculation():
    m=quality([(.75,1),(.25,0)])
    assert m['rmse']==pytest.approx(.25)
    assert m['log_loss']==pytest.approx(-math.log(.75))
    assert quality([(.2,.2),(.8,.8)])['rmse']==0


def test_calibration_fit_on_overconfident_scores():
    pairs=[(.9,.75),(.1,.25)]
    temperature=fit_temperature(pairs)
    assert temperature==2
    assert calibrate(.9,temperature)==pytest.approx(.75)
    assert calibrate(0,1)>0 and calibrate(1,1)<1


def test_interval_on_identical_cases():
    assert interval([.2]*8)=={'mean':pytest.approx(.2),'low':pytest.approx(.2),'high':pytest.approx(.2),'cases':8}


class TargetFixture:
    provenance={'backend':'test-fixture'}
    def __init__(self, queries):
        self.values=iter([q.target for q in queries])
    def score(self, request):
        p=next(self.values)
        return ScoreResult(request,'fixture',[1],[2,3],[math.log(p),math.log(1-p)],p,1,[2],0,self.provenance)


def test_report_keeps_calibration_separate(tmp_path):
    queries=synthetic_queries({'id':'fit','split':'calibration','counts':[16,4,8,72]})+synthetic_queries({'id':'test','split':'eval','counts':[8,12,24,56]})
    run_benchmark(queries, TargetFixture(queries), tmp_path/'run')
    report=summarize(tmp_path/'run')
    assert report['calibration']['temperature']==1
    assert report['calibration']['case_ids']==['fit']
    assert report['raw']['eval']['rmse']['mean']==pytest.approx(0)
    assert report['raw']['eval']['bayes_error']['mean']==pytest.approx(0)
    assert report['raw']['eval']['negation_error']['mean']==pytest.approx(0)
    assert report['decision']=='insufficient_cases'


def test_failed_observations_block_conclusions(tmp_path):
    class Broken:
        provenance={'backend':'test-fixture'}
        def score(self, request):
            raise ValueError('missing token')
    run_benchmark(synthetic_queries({'id':'fail','split':'eval','counts':[16,4,8,72]}), Broken(), tmp_path/'run')
    report=summarize(tmp_path/'run')
    assert report['decision']=='incomplete'
    assert len(report['errors'])==18
    assert 'calibrated' not in report


def test_evaluation_targets_do_not_change_temperature(tmp_path):
    from dataclasses import replace
    fit=synthetic_queries({'id':'fit','split':'calibration','counts':[16,4,8,72]})
    evaluation=synthetic_queries({'id':'test','split':'eval','counts':[8,12,24,56]})
    first=fit+evaluation
    second=fit+[replace(q,target=1-q.target) for q in evaluation]
    # Reuse identical observations while changing evaluation labels only.
    for name, queries in [('first',first),('second',second)]:
        run_benchmark(queries,TargetFixture(first),tmp_path/name)
    assert summarize(tmp_path/'first')['calibration']==summarize(tmp_path/'second')['calibration']


@pytest.mark.parametrize('field', ['request','provenance'])
def test_mismatched_observations_block_reporting(tmp_path,field):
    import json
    queries=synthetic_queries({'id':'fit','split':'calibration','counts':[16,4,8,72]})+synthetic_queries({'id':'test','split':'eval','counts':[8,12,24,56]})
    run=tmp_path/'run'
    run_benchmark(queries,TargetFixture(queries),run)
    p=run/'records.jsonl'
    records=[json.loads(line) for line in p.read_text().splitlines()]
    if field=='request':
        records[0]['result']['request']['proposition']='Another proposition.'
    else:
        records[0]['result']['provenance']['engine']={'model':'another-model'}
    p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    report=summarize(run)
    assert report['decision']=='incomplete'
    assert len(report['errors'])==1
    assert field in report['errors'][0]['error'].lower()
