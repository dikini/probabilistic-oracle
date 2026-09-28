import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('laya_bench',Path(__file__).parents[1]/'scripts/laya_oracle.py')
m=importlib.util.module_from_spec(spec)
if Path(spec.origin).exists():spec.loader.exec_module(m)


def test_case_counts_and_split_separation():
    rows=m.make_requests()
    assert len(rows)==210
    assert len({r['id'] for r in rows})==210
    cal={r['target'] for r in rows if r.get('split')=='calibration'}
    test={r['target'] for r in rows if r.get('split')=='test'}
    assert len(cal)==8 and len(test)==10 and cal.isdisjoint(test)


def test_semantic_remapping_ignores_label_and_position():
    rows=[r for r in m.make_requests() if r['id'].startswith('rate/calibration/0/') and r['question']['type']=='choice']
    assert len(rows)==4
    assert {tuple(r['question']['criteria']) for r in rows}=={('A','B'),('B','A')}
    for r in rows:
        p={k:0.8 if r['semantics'][k]=='positive' else 0.2 for k in r['question']['criteria']}
        assert m.positive_probability(r,p)==0.8


def test_distribution_rejects_missing_nonfinite_or_unnormalized():
    for probs in ([1], [float('nan'),0],[-.1,1.1],[.2,.2]):
        with pytest.raises(ValueError):m.distribution(probs,2)
    assert m.distribution([.2,.8],2)==[.2,.8]


def test_expected_losses_and_bayes():
    exact=m.metrics([.2,.8],[.2,.8])
    assert exact['rmse']==0
    assert exact['expected_brier']==pytest.approx(.16)
    assert m.posterior(.2)==pytest.approx(.5)
    assert m.posterior(0)==0 and m.posterior(1)==1
    assert m.fit_temperature([.2,.8],[.2,.8])==pytest.approx(1)


def test_logit_capture_agrees_with_native_response():
    q=dict(question=dict(type='choice',criteria={'B':'yes','A':'no'}),semantics={'B':'positive','A':'negative'})
    r=dict(logits=[0.,0.],shipped_temperature=1.,response={'answers':{'answer':{'choice':'B','probabilities':{'B':.5,'A':.5}}}})
    assert m.checked_probabilities(q,r)[1]=={'B':.5,'A':.5}
    import copy
    for logits in ([0.],[0.,0.,0.]):
        bad=copy.deepcopy(r);bad['logits']=logits
        with pytest.raises(ValueError,match='logit'):m.checked_probabilities(q,bad)
    bad=copy.deepcopy(r);bad['response']['answers']['answer']['probabilities']['B']=.6
    with pytest.raises(ValueError,match='SDK'):m.checked_probabilities(q,bad)
    q=dict(question=dict(type='noul'),semantics={'false':'negative','true':'positive'})
    r=dict(logits=[0.,0.],shipped_temperature=1.,response={'answers':{'answer':{'noul':.5}}})
    assert m.checked_probabilities(q,r)[1]['true']==.5
    r['response']['answers']['answer']['noul']=.7
    with pytest.raises(ValueError,match='SDK'):m.checked_probabilities(q,r)


def test_followup_corpus_counts_and_native_layout():
    rows=m.make_followup_requests()
    assert len(rows)==192
    weather=[r for r in rows if r['suite']=='weather']
    assert len(weather)==72
    assert all('Which category' in r['question']['instructions'] for r in weather)
    assert all('Which category' not in r['state'] for r in weather)
    for cohort,expected in [('low',4),('balanced',10),('high',16)]:
        for order in (0,1):
            group=[r for r in rows if r.get('cohort')==cohort and r['order']==order]
            assert len(group)==20 and sum(r['target']=='billing' for r in group)==expected
