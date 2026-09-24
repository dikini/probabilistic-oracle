import sys
from pathlib import Path
from collections import Counter
import importlib.util
import math
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('counter',Path(__file__).parents[1]/'scripts/counterbalanced_readout.py')
m=importlib.util.module_from_spec(spec)
if Path(spec.origin).exists():spec.loader.exec_module(m)

@pytest.mark.parametrize('n',[3,5])
def test_full_joint_balance(n):
    counts=Counter()
    for shift in range(n):
        for order in range(n):
            labels,display=m.layout(n,shift,order)
            for i in range(n):counts[i,labels[i],display.index(i)]+=1
    assert len(counts)==n**3 and set(counts.values())=={1}


def test_semantic_remapping():
    logs=m.semantic_logs([math.log(.6),math.log(.1),math.log(.2)],3,1)
    assert logs==pytest.approx([math.log(.1),math.log(.2),math.log(.6)])


def test_canonical_prompt_identical():
    from probabilistic_oracle.oracle import ScoreRequest
    import categorical_readout as c
    q=ScoreRequest('Context.','Proposition.')
    assert m.prompt_text(q,5,0,0)==c.prompt_text(q,5)


def test_always_a_is_identified_as_letter_bias():
    rows=[]
    for a in range(3):
        for b in range(3):rows.append(dict(mapping=a,order=b,letter_logprobs=[math.log(.7),math.log(.1),math.log(.1)]))
    d=m.diagnostics(rows,3)
    assert d['letter_counts']=={'A':9}
    assert d['position_counts']=={'1':3,'2':3,'3':3}
    assert d['category_counts']=={'TRUE':3,'NEITHER TRUE NOR FALSE':3,'FALSE':3}
    assert d['a_when_not_first']==dict(chosen=6,total=6)
    assert d['first_when_not_a']==dict(chosen=0,total=6)
    assert d['mean_raw_score']==pytest.approx(.5)


def test_incomplete_grid_rejected():
    with pytest.raises(ValueError):m.validate({'queries':[{'id':'x'}]},[])


def fixture_run():
    queries=m.make_queries(m.fresh_cases([]));frozen={str(n):dict(temperature=1,bias=0) for n in (3,5)}
    manifest=dict(queries=queries,query_sha256=m.c.digest(queries),protocol='test',protocol_sha256=m.c.digest('test'),
                  frozen_calibration=frozen,calibration_sha256=m.c.digest(frozen),fitted_constant=.43,backend={})
    by={q['id']:q for q in queries};records=[]
    for cell in m.grid(queries):
        n=cell['scale'];labels,display=m.layout(n,cell['mapping'],cell['order']);logs=[math.log(.8/n)]*n
        request=by[cell['id']]['request'];p,mass,_=m.c.score_logs(logs,m.c.anchors(n))
        records.append(dict(cell,status='ok',request=request,provenance={},category_letters=labels,display_order=display,
            candidate_token_ids=list(range(n)),letter_logprobs=logs,raw_score=p,candidate_mass=mass,generated_token_ids=[0],
            prompt_text=m.prompt_text(m.ScoreRequest(**request),n,cell['mapping'],cell['order'])))
    return manifest,records


def test_full_report_balances_and_keeps_controls_separate():
    manifest,records=fixture_run();r=m.summarize(manifest,records)
    assert r['total_requests']==476
    assert r['methods']['3']['synthetic_diagnostics']['count']==108
    assert r['methods']['5']['synthetic_diagnostics']['count']==300
    assert r['methods']['3']['per_query']['control-blue']['raw_balanced']==pytest.approx(.5)
    assert len(r['methods']['3']['accuracy']['raw_balanced']['world_rmse'])==4


@pytest.mark.parametrize('field,value',[('category_letters',['A']),('display_order',[0]),('letter_logprobs',[-1]),('raw_score',.99),('request',{}),('provenance',{'x':1}),('candidate_token_ids',[0,0]),('prompt_text','wrong')])
def test_tampering_fails(field,value):
    manifest,records=fixture_run();records[0][field]=value
    with pytest.raises(ValueError):m.validate(manifest,records)


def test_duplicate_and_failed_observations_fail_closed():
    manifest,records=fixture_run()
    with pytest.raises(ValueError):m.validate(manifest,records+[records[0]])
    records[0]['status']='error'
    with pytest.raises(ValueError):m.validate(manifest,records)


def test_freshness_excludes_previous_counts():
    cases=m.fresh_cases([]);assert cases==m.fresh_cases([])
    new=m.fresh_cases([x['counts'] for x in cases])
    assert not {tuple(x['counts']) for x in cases}&{tuple(x['counts']) for x in new}
