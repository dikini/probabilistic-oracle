import importlib.util
from pathlib import Path
import pytest
import math

spec=importlib.util.spec_from_file_location('compare_readouts',Path(__file__).parents[1]/'scripts/compare_readouts.py')
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.mark.parametrize('text,expected',[('0',0),('1',1),(' 0.25\n',.25),('.5',.5),('1e-1',.1)])
def test_numeric_answer_parser(text,expected):
    assert comparison.parse_probability(text,'stop')==expected


@pytest.mark.parametrize('text,reason',[('25%','stop'),('1/4','stop'),('Probability: 0.25','stop'),('nan','stop'),('1.1','stop'),('-0.1','stop'),('','stop'),('0.25','length')])
def test_invalid_answers_are_not_repaired(text,reason):
    with pytest.raises(ValueError):
        comparison.parse_probability(text,reason)


def test_new_worlds_exclude_every_previous_table():
    excluded=[[16,4,8,72]]
    cases=comparison.fresh_cases(excluded)
    assert cases==comparison.fresh_cases(excluded)
    assert len(cases)==len({tuple(c['counts']) for c in cases})==8
    assert all(c['counts'] not in excluded and sum(c['counts'])==100 for c in cases)


def test_summary_detects_readout_advantage_and_errors():
    from dataclasses import asdict
    from probabilistic_oracle.bench import synthetic_queries
    cases=[{'id':f'case-{i}','split':'eval','counts':[16,4,8,72]} for i in range(8)]
    queries=[asdict(q) for c in cases for q in synthetic_queries(c,('A','B'))]
    observations=[]
    for q in queries:
        observations.append({'id':q['id'],'numeric':{'status':'ok','probability':q['target'],'text':str(q['target']),'finish_reason':'stop','request':q['request']},
                             'binary':{'status':'ok','raw_score':.5,'calibrated_score':comparison.calibrate_binary(.5),'observation':{'request':q['request'],'raw_logprobs':[math.log(.5)]*2,'score':.5,'provenance':{'backend':'test-fixture'}}}})
    result=comparison.summarize(queries,observations)
    assert result['decision']=='numeric_readout_supported'
    assert result['numeric']['eval']['rmse']['mean']==0
    assert result['paired_rmse_difference']['high']<0
    observations[0]['numeric']={'status':'error','error':'Malformed numerical output'}
    result=comparison.summarize(queries,observations)
    assert result['decision']=='incomplete'
    assert result['coverage']['numeric_valid']==143
    assert result['excluded_cases']==['case-0']
    assert result['paired_cases']==7


def test_missing_pair_cannot_be_reported_as_complete():
    with pytest.raises(ValueError,match='Incomplete'):
        comparison.summarize([{'id':'expected'}],[])


def test_numeric_prompt_preserves_conditioning_sections():
    from probabilistic_oracle.oracle import ScoreRequest
    class InspectTokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs['enable_thinking'] is False
            return messages[0]['content']
    request=ScoreRequest('The exact context.','The exact proposition.',assumptions=('Assume H.',),observations=('Observed E.',),verbalizers=('A','B'))
    text=comparison.numerical_prompt(InspectTokenizer(),request)
    for fragment in ['Context: The exact context.', 'Assumptions (suppose these hold):\nAssume H.',
                     'Observations (new evidence):\nObserved E.', 'Proposition: The exact proposition.']:
        assert fragment in text


def test_frozen_binary_calibration():
    assert comparison.calibrate_binary(.5)==pytest.approx(.3775406688)


@pytest.mark.parametrize('mutation', ['numeric_text','binary_calibration','numeric_request','binary_request','binary_logs'])
def test_stale_derived_scores_do_not_override_raw_observations(mutation):
    from dataclasses import asdict
    from probabilistic_oracle.bench import synthetic_queries
    import copy
    queries=[asdict(q) for q in synthetic_queries({'id':'one','split':'eval','counts':[16,4,8,72]})]
    q=queries[0]
    row={'id':q['id'],'numeric':{'status':'ok','text':'.2','finish_reason':'stop','probability':.2,'request':dict(q['request'])},
         'binary':{'status':'ok','raw_score':.5,'calibrated_score':comparison.calibrate_binary(.5),
                   'observation':{'request':dict(q['request']),'raw_logprobs':[math.log(.5)]*2,'score':.5,'provenance':{'backend':'test-fixture'}}}}
    observations=[]
    for query in queries:
        item=copy.deepcopy(row)
        item['id']=query['id']
        item['numeric']['request']=dict(query['request'])
        item['binary']['observation']['request']=dict(query['request'])
        observations.append(item)
    row=observations[0]
    if mutation=='numeric_text':
        row['numeric'].update(text='20%',finish_reason='length')
    elif mutation=='binary_calibration':
        row['binary']['calibrated_score']=.5
    elif mutation=='numeric_request':
        row['numeric']['request']['context']='Other context'
    elif mutation=='binary_request':
        row['binary']['observation']['request']['context']='Other context'
    else:
        row['binary']['observation']['raw_logprobs']=[math.log(.8),math.log(.2)]
    result=comparison.summarize(queries,observations)
    assert result['decision']=='incomplete'
    assert len(result['errors'])==1


def test_saved_report_rejects_changed_query_manifest(tmp_path):
    import json
    (tmp_path/'manifest.json').write_text(json.dumps({'queries':[],'query_sha256':'incorrect'}))
    with pytest.raises(ValueError,match='hash'):
        comparison.report_saved(tmp_path)
