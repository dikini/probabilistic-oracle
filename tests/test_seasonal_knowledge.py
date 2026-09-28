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
