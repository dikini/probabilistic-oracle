import json
from pathlib import Path
from probabilistic_oracle.cli import main
from probabilistic_oracle.bench import run_benchmark, synthetic_queries


def test_validate_cases_without_gpu(capsys):
    assert main(['validate'])==0
    out=json.loads(capsys.readouterr().out)
    assert out['queries']==240
    assert out['cases_by_split']=={'calibration':4,'eval':8,'world':4}


def test_report_failed_run_is_explicit(tmp_path,capsys):
    class Broken:
        provenance={'backend':'test-fixture'}
        def score(self, request):
            raise ValueError('deliberate failure')
    run=tmp_path/'run'
    run_benchmark(synthetic_queries({'id':'case','split':'eval','counts':[1,1,1,1]}),Broken(),run)
    assert main(['report',str(run)])==2
    out=json.loads(capsys.readouterr().out)
    assert out['decision']=='incomplete'
    assert json.loads((run/'report.json').read_text())==out


def test_refuses_existing_run_before_gpu_init(tmp_path,capsys):
    assert main(['run','--output',str(tmp_path)])==2
    assert 'already exists' in capsys.readouterr().err
