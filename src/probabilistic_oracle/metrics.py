"""Calibration fitted on separate worlds and descriptive case-level metrics."""

import math
import random
from statistics import mean, pvariance

from .algebra import probability, bayes, UndefinedConditioning
from .bench import read_run
from .oracle import normalize

TEMPERATURES = (.25, .5, .75, 1., 1.5, 2., 3., 4., 8.)


def _clip(p):
    return min(1-1e-12, max(1e-12, probability(p)))


def quality(pairs):
    if not pairs:
        raise ValueError('No probability targets')
    pairs = [(probability(p), probability(y)) for p, y in pairs]
    return dict(rmse=math.sqrt(mean((p-y)**2 for p, y in pairs)),
                log_loss=mean(-y*math.log(_clip(p))-(1-y)*math.log1p(-_clip(p)) for p, y in pairs))


def calibrate(p, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    p = _clip(p)
    log_odds = (math.log(p)-math.log1p(-p))/temperature
    return 1/(1+math.exp(-log_odds)) if log_odds >= 0 else math.exp(log_odds)/(1+math.exp(log_odds))


def fit_temperature(pairs):
    if not pairs:
        raise ValueError('Calibration split has no base queries')
    return min(TEMPERATURES, key=lambda t: (quality([(calibrate(p,t), y) for p,y in pairs])['log_loss'], abs(t-1)))


def interval(values):
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('Intervals require finite case metrics')
    rng = random.Random(0)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(500))
    return dict(mean=mean(values), low=draws[12], high=draws[487], cases=len(values))


def _case_metrics(rows, transform):
    scores = {(q['variant'],q['kind']): transform(p) for q,p in rows}
    base = {kind:p for (variant,kind),p in scores.items() if variant == 'base'}
    synthetic = rows[0][0]['split'] != 'world'
    expected = {'h','not_h','e_h','e_not_h','posterior','joint'} if synthetic else {'h','not_h'}
    if set(scores) != {(v,k) for v in ('base','paraphrase','irrelevant') for k in expected}:
        raise ValueError('Incomplete query family')
    result = dict(
        negation_error=abs(base['h']+base['not_h']-1),
        paraphrase_mae=mean(abs(scores['paraphrase', k]-p) for k,p in base.items()),
        paraphrase_variance=mean(pvariance([p,scores['paraphrase', k]]) for k,p in base.items()),
        irrelevant_context_mae=mean(abs(scores['irrelevant', k]-p) for k,p in base.items()),
    )
    if synthetic:
        result.update(quality([(transform(p), q['target']) for q,p in rows if q['variant']=='base']))
        result['conjunction_error'] = abs(base['joint']-base['h']*base['e_h'])
        try:
            result['bayes_error'] = abs(base['posterior']-bayes(base['h'],base['e_h'],base['e_not_h']))
        except UndefinedConditioning:
            result['bayes_error'] = None
    return result


def _aggregate(groups, transform):
    out = {}
    for split in ('eval', 'world'):
        cases = [_case_metrics(rows, transform) for rows in groups.values() if rows[0][0]['split']==split]
        metrics = {}
        if cases:
            for key in cases[0]:
                valid = [c[key] for c in cases if c[key] is not None]
                metrics[key] = interval(valid) if valid else None
            metrics['undefined_bayes_paths'] = sum(c.get('bayes_error', 0) is None for c in cases)
        out[split] = metrics
    return out


def summarize(directory):
    manifest, records = read_run(directory)
    queries = {q['id']:q for q in manifest['queries']}
    report = dict(schema_version=1, query_sha256=manifest['query_sha256'],
                  backend=manifest['backend'], metadata=manifest['metadata'],
                  total_queries=len(records), errors=[])
    groups = {}
    for record in records:
        if record['status']=='error':
            report['errors'].append(dict(id=record['id'], error=record['error']))
            continue
        q = queries[record['id']]
        result = record['result']
        try:
            p = probability(result['score'])
            raw_score, _ = normalize(*result['raw_logprobs'])
            if not math.isclose(p,raw_score,abs_tol=1e-10):
                raise ValueError('Stored score does not match raw observations')
            if result['request'] != q['request']:
                raise ValueError('Result request does not match the query manifest')
            if result['provenance'] != manifest['backend']:
                raise ValueError('Result provenance does not match the run manifest')
            if q['split'] not in ('calibration','eval','world'):
                raise ValueError('Unknown split')
            groups.setdefault(q['case_id'], []).append((q,p))
        except (ValueError, KeyError, TypeError) as exc:
            report['errors'].append(dict(id=record['id'],error=str(exc)))
    if report['errors']:
        report['decision']='incomplete'
        return report
    if any(len({q['split'] for q,p in rows}) != 1 for rows in groups.values()):
        raise ValueError('Case leaks across splits')
    fit_ids = sorted(case for case,rows in groups.items() if rows[0][0]['split']=='calibration')
    fit = [(p,q['target']) for case in fit_ids for q,p in groups[case] if q['variant']=='base']
    # Validate every family, including calibration variants not used for fitting.
    for rows in groups.values():
        _case_metrics(rows, lambda p:p)
    if not fit or not any(rows[0][0]['split']=='eval' for rows in groups.values()):
        report['decision']='insufficient_cases'
        return report
    temperature = fit_temperature(fit)
    report['calibration'] = dict(method='binary-log-odds-temperature-grid', temperature=temperature,
                                 grid=TEMPERATURES, case_ids=fit_ids, query_count=len(fit),
                                 fit_variant='base', objective='soft_target_log_loss')
    report['raw'] = _aggregate(groups, lambda p:p)
    report['calibrated'] = _aggregate(groups, lambda p:calibrate(p,temperature))
    fitted_constant = mean(y for p,y in fit)
    report['baselines'] = {}
    for name, constant in [('constant_half', .5), ('calibration_mean', fitted_constant)]:
        values = [quality([(constant,q['target']) for q,p in rows if q['variant']=='base'])['rmse']
                  for rows in groups.values() if rows[0][0]['split']=='eval']
        report['baselines'][name] = dict(score=constant, rmse=interval(values))
    eval_count=sum(rows[0][0]['split']=='eval' for rows in groups.values())
    checks = report['calibrated']['eval']
    gates = {k: checks[k]['mean'] <= threshold for k,threshold in {
        'rmse':.1,'negation_error':.1,'conjunction_error':.1,'bayes_error':.1,
        'paraphrase_mae':.05,'irrelevant_context_mae':.05}.items()}
    gates['beats_baselines'] = all(checks['rmse']['mean'] < b['rmse']['mean'] for b in report['baselines'].values())
    gates['defined_bayes_paths'] = checks['undefined_bayes_paths']==0 and report['raw']['eval']['undefined_bayes_paths']==0
    report['gates']=gates
    if len(fit_ids)<4 or eval_count<8:
        report['decision']='insufficient_cases'
    elif manifest['backend']['backend'] != 'vllm':
        report['decision']='non_model_data'
    else:
        report['decision']='thresholds_met' if all(gates.values()) else 'thresholds_not_met'
    return report
