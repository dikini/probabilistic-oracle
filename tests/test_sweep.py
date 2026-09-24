import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('parameter_sweep',Path(__file__).parents[1]/'scripts/parameter_sweep.py')
sweep=importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_affine_probability_transform():
    assert sweep.transform(.9,2,0)==pytest.approx(.75)
    assert sweep.transform(.5,1,0)==.5
    assert sweep.transform(.5,1,1)==pytest.approx(.7310585786)


def test_fit_ignores_development_targets():
    rows=[({'split':'calibration','variant':'base','target':.75},.9),
          ({'split':'calibration','variant':'base','target':.25},.1),
          ({'split':'eval','variant':'base','target':0},.99)]
    rows=rows[:2]*12+rows[2:]
    first=sweep.fit(rows)
    rows[-1][0]['target']=1
    assert sweep.fit(rows)==first
    assert first['temperature']==2
    assert first['bias']==0


def test_holdout_is_fresh_reproducible_and_well_formed():
    existing=[[16,4,8,72]]
    a=sweep.fresh_worlds(existing)
    assert a==sweep.fresh_worlds(existing)
    assert len(a)==8
    assert len({tuple(c['counts']) for c in a})==8
    assert all(sum(c['counts'])==100 and min(c['counts'])>0 and c['counts'] not in existing for c in a)
