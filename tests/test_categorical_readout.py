import importlib.util
import math
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('categorical',Path(__file__).parents[1]/'scripts/categorical_readout.py')
m=importlib.util.module_from_spec(spec)
if spec.loader and Path(spec.origin).exists(): spec.loader.exec_module(m)


def test_weighted_score_and_mass():
    score,mass,weights=m.score_logs([math.log(.1),math.log(.2),math.log(.1)],[1,.5,0])
    assert score==pytest.approx(.5)
    assert mass==pytest.approx(.4)
    assert weights==pytest.approx([.25,.5,.25])


def test_binary_reduces_to_log_odds_temperature_and_bias():
    p,_,_=m.score_logs([math.log(.8),math.log(.2)],[1,0],2,-.5)
    assert p==pytest.approx(1/(1+math.exp(-(math.log(4)/2-.5))))


@pytest.mark.parametrize('logs,anchors', [([-1],[1,0]),([float('nan'),-1],[1,0]),([0,0],[1,0]),([-1,-2],[1,2]),([-1,None],[1,0])])
def test_invalid_observations_rejected(logs,anchors):
    with pytest.raises(ValueError): m.score_logs(logs,anchors)


def test_missing_candidate_is_not_zero_filled():
    with pytest.raises(ValueError): m.extract({1:-1},[1,2],[1,0])


def test_prompt_has_categories_without_numerical_request():
    from probabilistic_oracle.oracle import ScoreRequest
    text=m.prompt_text(ScoreRequest('Some context.','Some proposition.'),5)
    assert 'MOSTLY TRUE' in text and 'NEITHER TRUE NOR FALSE' in text
    assert 'probability' not in text.lower() and '0.5' not in text


def test_fresh_worlds_are_disjoint_and_reproducible():
    cases=m.fresh_cases([])
    assert cases==m.fresh_cases([])
    assert len(cases)==12 and len({tuple(c['counts']) for c in cases})==12
    assert [c['split'] for c in cases].count('calibration')==4
    assert not {tuple(c['counts']) for c in cases}&{tuple(c['counts']) for c in m.fresh_cases([c['counts'] for c in cases])}


def test_fit_does_not_consume_evaluation_targets():
    rows=[({'split':'calibration','variant':'base','target':.1},[-3,-2,-1.5]),
          ({'split':'eval','variant':'base','target':.9},[-1,-2,-3])]
    before=m.fit(rows,[1,.5,0])
    rows[1][0]['target']=0
    assert m.fit(rows,[1,.5,0])==before


def test_fit_requires_calibration():
    with pytest.raises(ValueError):m.fit([], [1,0])


def test_incomplete_records_rejected():
    with pytest.raises(ValueError): m.validate_records({'queries':[{'id':'one'}]},[])


def fixture_run():
    from dataclasses import asdict
    import json
    qs=[asdict(q) for c in m.fresh_cases([]) for q in m.synthetic_queries(c,('A','B')) if q.split=='eval' or q.variant=='base']
    qs=json.loads(json.dumps(qs))
    manifest=dict(queries=qs,query_sha256=m.digest(qs),protocol='test',protocol_sha256=m.digest('test'),backend={})
    records=[]
    for q in qs:
        for n in m.CATEGORIES:
            logs=[math.log(.8/n)]*n;p,mass,_=m.score_logs(logs,m.anchors(n))
            records.append(dict(id=q['id'],scale=n,status='ok',request=q['request'],provenance={},
                                candidate_token_ids=list(range(n)),raw_logprobs=logs,raw_score=p,candidate_mass=mass,
                                categories=m.CATEGORIES[n],anchors=m.anchors(n),generated_token_ids=[0]))
    frozen={str(n):m.fit([(q,[math.log(.8/n)]*n) for q in qs],m.anchors(n)) for n in m.CATEGORIES}
    return manifest,records,frozen


def test_complete_report_and_no_false_success():
    manifest,records,frozen=fixture_run()
    result=m.summarize(manifest,records,frozen)
    assert result['total_requests']==504
    assert result['decision']=='thresholds_not_met'
    assert set(result['paired_differences'])=={'3','5'}


@pytest.mark.parametrize('field,value', [('raw_score',.99),('candidate_mass',.2),('request',{}),('categories',['oops']),('provenance',{'oops':True}),('raw_logprobs',[-1]),('candidate_token_ids',[0,0])])
def test_tampering_rejected(field,value):
    manifest,records,_=fixture_run();records[0][field]=value
    with pytest.raises(ValueError):m.validate_records(manifest,records)


def test_frozen_calibration_and_manifest_checked():
    manifest,records,frozen=fixture_run()
    frozen['3']['temperature']=999
    with pytest.raises(ValueError,match='frozen'):m.summarize(manifest,records,frozen)
    manifest['query_sha256']='wrong'
    with pytest.raises(ValueError,match='hash'):m.summarize(manifest,records,frozen)
