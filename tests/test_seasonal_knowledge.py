import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('seasonal',Path(__file__).parents[1]/'scripts/seasonal_knowledge.py')
m=importlib.util.module_from_spec(spec)
if Path(spec.origin).exists():spec.loader.exec_module(m)

@pytest.mark.parametrize('text,reason',[('hot.','stop'),('hot','length'),('','stop'),('HOT','stop'),(' hot explanation','stop')])
def test_invalid_answers_rejected(text,reason):
    with pytest.raises(ValueError):m.parse(text,reason)


def test_valid_word_and_unknown():
    assert m.parse(' hot\n','stop')=='hot'
    assert m.parse('unknown','stop')=='unknown'


def test_prompts_do_not_contain_reference_answers():
    cases=m.load_cases();prompts=m.queries(cases)
    assert len(prompts)==36
    assert len({p['id'] for p in prompts})==36
    for p in prompts:
        assert 'http' not in p['text'] and 'probability' not in p['text']
    assert 'BULGARIA' in next(p['text'] for p in prompts if p['id']=='bulgaria-january/emphasis')


def test_stable_wrong_is_not_knowledge_success():
    qs=m.queries(m.load_cases());rows=[dict(id=q['id'],status='ok',text='cold',finish_reason='stop',answer='cold') for q in qs]
    r=m.summarize(qs,rows)
    assert r['strict']['correct']==9 and r['strict']['total']==15
    assert r['stable_cases']==12
    assert r['singapore_hot_pairs']==0
    assert r['seasonal_counts']=={'tied':15}


def test_missing_and_duplicate_records_rejected():
    qs=m.queries(m.load_cases())
    with pytest.raises(ValueError):m.summarize(qs,[])
    rows=[dict(id=q['id'],status='ok',text='cold',finish_reason='stop',answer='cold') for q in qs]
    with pytest.raises(ValueError):m.summarize(qs,rows+[rows[0]])


def test_unknown_does_not_become_temperature_or_correct_label():
    qs=m.queries(m.load_cases());rows=[dict(id=q['id'],status='ok',text='unknown',finish_reason='stop',answer='unknown') for q in qs]
    r=m.summarize(qs,rows)
    assert r['strict']['correct']==0 and r['unknown_count']==36
    assert r['seasonal_counts']=={'unknown':15}


def test_error_produces_incomplete_report():
    qs=m.queries(m.load_cases());rows=[dict(id=q['id'],status='error',error='test failure') for q in qs]
    assert m.summarize(qs,rows)['status']=='incomplete'


def test_expected_and_reversed_seasonal_contrasts():
    qs=m.queries(m.load_cases());rows=[]
    for q in qs:
        c=q['case']
        answer='hot' if c['country']=='Singapore' or c['month']==c['warmer_month'] else 'cold'
        rows.append(dict(id=q['id'],status='ok',text=answer,finish_reason='stop',answer=answer))
    r=m.summarize(qs,rows)
    assert r['seasonal_counts']=={'expected':15} and r['singapore_hot_pairs']==3
    for row in rows:
        if not row['id'].startswith('singapore'):
            row['text']=row['answer']='cold' if row['answer']=='hot' else 'hot'
    assert m.summarize(qs,rows)['seasonal_counts']=={'reversed':15}


def test_stored_answer_cannot_override_raw_output():
    qs=m.queries(m.load_cases());rows=[dict(id=q['id'],status='ok',text='hot',finish_reason='stop',answer='cold') for q in qs]
    assert m.summarize(qs,rows)['status']=='incomplete'


def test_supplied_facts_preserve_original_prompt_as_suffix():
    cases=m.load_cases();facts={c['country']:dict(text='Fact about '+c['country']) for c in cases}
    old=m.queries(cases);new=m.queries(cases,facts)
    for a,b in zip(old,new):
        assert b['text'].endswith(a['text'])
        assert b['text'].startswith('Background facts for this question (take as given):\nFact about '+a['case']['country']+'\n\n')
        assert a['id']==b['id'] and a['case']==b['case']


def test_missing_facts_rejected():
    with pytest.raises(ValueError):m.queries(m.load_cases(),{})


def test_comparison_counts_changes_without_changing_labels():
    qs=m.queries(m.load_cases())
    a=m.summarize(qs,[dict(id=q['id'],status='ok',text='unknown',finish_reason='stop',answer='unknown') for q in qs])
    b=m.summarize(qs,[dict(id=q['id'],status='ok',text='cold',finish_reason='stop',answer='cold') for q in qs])
    result=m.compare_reports(a,b)
    assert result['changed_answers']==36 and result['strict_correct_delta']==9


def test_provided_labels_scored_separately_from_climate_labels():
    import json
    facts=json.loads((m.ROOT/'bench/cases/seasonal-facts.json').read_text())
    qs=m.queries(m.load_cases(),facts)
    rows=[dict(id=q['id'],status='ok',text=q['provided_label'],finish_reason='stop',answer=q['provided_label']) for q in qs]
    r=m.summarize(qs,rows)
    assert r['supplied_knowledge_agreement']==dict(correct=36,total=36)
    assert r['strict']==dict(correct=15,total=15)
    assert sum(q['case']['ambiguous'] for q in qs)==21


def test_baseline_requires_same_model_and_sampling():
    from probabilistic_oracle.backends.vllm import engine_config
    small = dict(engine=engine_config(), sampling=dict(temperature=0))
    large = dict(engine=engine_config('qwen3-4b-awq'), sampling=dict(temperature=0))
    assert m.validate_baseline_backend(dict(backend=large), large['engine'], large['sampling']) == large
    with pytest.raises(ValueError, match='Baseline backend'):
        m.validate_baseline_backend(dict(backend=small), large['engine'], large['sampling'])
    with pytest.raises(ValueError, match='Baseline backend'):
        m.validate_baseline_backend(dict(backend=large), large['engine'], dict(temperature=1))
