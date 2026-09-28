import importlib.util
from pathlib import Path
import math
import pytest
spec=importlib.util.spec_from_file_location('diagnosis',Path(__file__).parents[1]/'scripts/frozen_diagnosis.py')
m=importlib.util.module_from_spec(spec)
if Path(spec.origin).exists():spec.loader.exec_module(m)


def test_worlds_share_marginals_but_not_joint():
    a,b=m.worlds().values()
    for h in m.FAULTS:
        assert sum(a['joint'][h])==pytest.approx(1)
        assert sum(b['joint'][h])==pytest.approx(1)
        assert min(b['joint'][h])>0
        assert m.marginals(a['joint'][h])==pytest.approx(m.marginals(b['joint'][h]))
    assert a['joint']!=b['joint']


def test_naive_equals_truth_only_in_independent_world():
    a,b=m.worlds().values()
    for pattern in m.PATTERNS:
        assert m.truth(a,pattern)==pytest.approx(m.naive(a,pattern))
    assert max(abs(x-y) for x,y in zip(m.truth(b,'01'),m.naive(b,'01')))>0.2
    assert sum(m.evidence_mass(b,p) for p in m.PATTERNS)==pytest.approx(1)


def test_request_counts_and_reduced_information_identical():
    requests=m.make_requests();assert len(requests)==324
    by={q['id']:q for q in requests};assert len(by)==324
    for q in requests:
        if q.get('coverage')=='reduced':
            other=by[q['id'].replace('/independent/','/correlated/')]
            assert q['state']==other['state'] and q['question']==other['question']
            assert '00=' not in q['state'] and '11=' not in q['state']
            assert 'Joint symptom probabilities are not supplied.' in q['state']
    params=[q for q in requests if q['role'] in ('prior','joint','x','y')]
    assert all('Observed report:' not in q['state'] for q in params)
    assert not any(q['role']=='joint' and q['coverage']=='reduced' for q in requests)


def test_scores_use_distribution_not_argmax_truth():
    assert m.scores([.5,.3,.2],[.5,.3,.2])['accuracy']==.5
    assert m.scores([.5,.3,.2],[.5,.3,.2])['log_loss']==pytest.approx(-sum(p*math.log(p) for p in [.5,.3,.2]))
    assert m.scores([1,0,0],[1,0,0])['brier']==0


def test_missing_observations_rejected(tmp_path):
    m.base.write(tmp_path/'manifest.json',dict(requests=m.make_requests(),requests_sha256=m.base.digest(m.make_requests()),worlds=m.worlds()))
    (tmp_path/'records.jsonl').write_text('')
    with pytest.raises(ValueError,match='Incomplete'):m.report_saved(tmp_path)


def test_saved_world_definition_cannot_silently_change(tmp_path):
    import copy
    for component in ('prior','joint'):
        changed=copy.deepcopy(m.worlds())
        if component=='prior':changed['independent']['prior']=[.4,.3,.3]
        else:changed['independent']['joint']['coolant']=[.1,.1,.2,.6]
        requests=m.make_requests()
        m.base.write(tmp_path/'manifest.json',dict(requests=requests,requests_sha256=m.base.digest(requests),worlds=changed))
        (tmp_path/'records.jsonl').write_text('')
        with pytest.raises(ValueError,match='World definition'):m.report_saved(tmp_path)


def test_perfect_parameter_records_recover_exact_joint_reference(tmp_path):
    """Synthetic CPU fixture, not observations from a model."""
    import json
    requests=m.make_requests();records=[]
    for q in requests:
        role=q['role'];keys=list(q['question']['criteria'])
        if role=='extract':values={k:1-3e-8 if k==q['pattern'] else 1e-8 for k in keys}
        else:
            world=m.worlds()[q['world']]
            if role=='prior':values=dict(zip(m.FAULTS,world['prior']))
            elif role=='joint':values=dict(zip(m.PATTERNS,world['joint'][q['fault']]))
            elif role in ('x','y'):
                p=m.marginals(world['joint'][q['fault']])[0 if role=='x' else 1];values={'present':p,'absent':1-p}
            else:values=dict(zip(m.FAULTS,m.truth(world,q['pattern'])))
        records.append(dict(id=q['id'],request=q,status='ok',seconds=.01,logits=[math.log(values[k]) for k in keys],shipped_temperature=1.,
            response={'answers':{'answer':{'choice':max(keys,key=lambda k:values[k]),'probabilities':{k:round(values[k],4) for k in keys}}}}))
    m.base.write(tmp_path/'manifest.json',dict(requests=requests,requests_sha256=m.base.digest(requests),worlds=m.worlds()))
    (tmp_path/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    report=m.report_saved(tmp_path)['readouts']['shipped']['summary']
    for world in m.worlds():
        truth=report[f'{world}/full/reference_joint']
        for method in ('direct','factorized_joint','hybrid_joint'):
            for metric in ('accuracy','log_loss','brier'):
                assert report[f'{world}/full/{method}'][metric]==pytest.approx(truth[metric])
        assert report[f'{world}/full/factorized_joint']['unique_requests']==48
        assert report[f'{world}/full/factorized_naive']['unique_requests']==66
        assert report[f'{world}/full/hybrid_joint']['unique_requests']==24
        assert f'{world}/reduced/hybrid_joint' not in report
