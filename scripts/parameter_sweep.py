"""Time-bounded experimental sweep; production scoring defaults stay unchanged."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import time

from probabilistic_oracle.bench import load_queries, read_run, run_benchmark, synthetic_queries
from probabilistic_oracle.metrics import _aggregate, quality
from probabilistic_oracle.oracle import normalize

TEMPERATURES = (.125,.25,.5,.75,1,1.5,2,3,4,8,16,32)
BIASES = (-2,-1,-.5,-.25,0,.25,.5,1,2)
LIMITS = dict(rmse=.1,negation_error=.1,conjunction_error=.1,bayes_error=.1,
              paraphrase_mae=.05,irrelevant_context_mae=.05)
PROMPTS = {
    'baseline': '',
    'sample': (
        'Imagine sampling one actual outcome uniformly from the possibilities described below. '
        'Choose the answer that describes that sampled outcome, not whether the proposition is possible '
        'or logically guaranteed. Assumptions and observations restrict the sample space. '
        'Give only the answer label, without explanation.\n\n'),
    'frequency': (
        'Act as a probabilistic binary oracle. Your relative likelihoods for the two answer labels '
        'should match the fraction of outcomes where the proposition is true versus false. '
        'Use the stated counts and uniform selection. Condition on all assumptions and observations. '
        'Do not turn an uncertain event into a categorical certainty judgment. Output one label only.\n\n'),
    'examples': (
        'Act as a probabilistic binary oracle. Match the relative answer-label likelihoods to the '
        'proposition probability. For a fair coin, heads has true:false odds 1:1. For one uniformly '
        'chosen item from three red and one blue item, red has odds 3:1 and blue has odds 1:3. '
        'Under a condition, restrict to outcomes satisfying it before assessing the proposition. '
        'Use the actual counts below, not the example counts. Output one label only.\n\n'),
}
CONFIGS = [
    dict(name='baseline-yes-no',style='baseline',labels=['YES','NO']),
    dict(name='baseline-a-b',style='baseline',labels=['A','B']),
    dict(name='sample-true-false',style='sample',labels=['TRUE','FALSE']),
    dict(name='frequency-true-false',style='frequency',labels=['TRUE','FALSE']),
    dict(name='frequency-a-b',style='frequency',labels=['A','B']),
    dict(name='examples-true-false',style='examples',labels=['TRUE','FALSE']),
]


def transform(p, temperature, bias):
    if not math.isfinite(p) or not 0 <= p <= 1 or temperature <= 0:
        raise ValueError('Invalid calibration parameters')
    p=min(1-1e-12,max(1e-12,p))
    z=(math.log(p)-math.log1p(-p))/temperature+bias
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))


def fit(rows, biases=BIASES):
    pairs=[(p,q['target']) for q,p in rows if q['split']=='calibration' and q['variant']=='base']
    if len(pairs)!=24:
        raise ValueError('Expected 24 calibration base queries')
    def loss(t,b):
        return quality([(transform(p,t,b),y) for p,y in pairs])['log_loss']
    t,b=min(((t,b) for t in TEMPERATURES for b in biases),key=lambda tb:(loss(*tb),abs(tb[1]),abs(tb[0]-1)))
    return dict(temperature=t,bias=b,fit_log_loss=loss(t,b),fit_query_count=len(pairs),
                fitted_constant=mean(y for p,y in pairs))


def fresh_worlds(existing):
    rng=random.Random(240924)
    seen={tuple(c) for c in existing}
    worlds=[]
    while len(worlds)<8:
        cuts=[0]+sorted(rng.sample(range(1,100),3))+[100]
        counts=[b-a for a,b in zip(cuts,cuts[1:])]
        if tuple(counts) in seen:
            continue
        seen.add(tuple(counts))
        worlds.append(dict(id=f"fresh-{len(worlds)+1:02}",split='eval',counts=counts))
    return worlds


def load_rows(path):
    manifest,records=read_run(path)
    queries={q['id']:q for q in manifest['queries']}
    rows=[]
    for r in records:
        if r['status']!='ok':
            raise ValueError(f"Failed observation {r['id']}: {r.get('error')}")
        q=queries[r['id']]; result=r['result']
        if result['request']!=q['request'] or result['provenance']!=manifest['backend']:
            raise ValueError('Request/provenance mismatch')
        expected,_=normalize(*result['raw_logprobs'])
        if not math.isclose(expected,result['score'],abs_tol=1e-10):
            raise ValueError('Score/log-probability mismatch')
        rows.append((q,result['score']))
    return rows


def measurements(rows, calibration=None, fitted_constant=None):
    groups={}
    for q,p in rows:
        if q['split']!='calibration':
            groups.setdefault(q['case_id'],[]).append((q,p))
    fn=(lambda p:p) if calibration is None else (lambda p:transform(p,calibration['temperature'],calibration['bias']))
    metrics=_aggregate(groups,fn)
    targets=[q['target'] for q,p in rows if q['split']=='calibration' and q['variant']=='base']
    fitted=fitted_constant if fitted_constant is not None else (calibration['fitted_constant'] if calibration else mean(targets))
    metrics['baselines']={}
    for name,constant in [('constant_half',.5),('calibration_mean',fitted)]:
        rmses=[quality([(constant,q['target']) for q,p in group if q['variant']=='base'])['rmse']
               for group in groups.values() if group[0][0]['split']=='eval']
        metrics['baselines'][name]=dict(score=constant,rmse=mean(rmses))
    return metrics


def gates(metrics):
    scores=metrics['eval']
    checks={k:scores[k] is not None and scores[k]['mean']<=limit for k,limit in LIMITS.items()}
    checks['beats_baselines']=all(scores['rmse']['mean']<x['rmse'] for x in metrics['baselines'].values())
    checks['defined_bayes_paths']=scores['undefined_bayes_paths']==0
    return checks


def analyze(rows,config):
    affine=fit(rows)
    temperature_only=fit(rows,biases=(0,))
    calibrated=measurements(rows,affine)
    return dict(config=config,calibration=affine,raw=measurements(rows),
                temperature_only=dict(calibration=temperature_only,metrics=measurements(rows,temperature_only)),
                calibrated=calibrated,gates=gates(calibrated))


def rank(result):
    metrics=result['calibrated']['eval']
    if metrics['undefined_bayes_paths']:
        return math.inf,math.inf
    ratios=[metrics[k]['mean']/limit for k,limit in LIMITS.items()]
    baseline=min(x['rmse'] for x in result['calibrated']['baselines'].values())
    ratios.append(metrics['rmse']['mean']/baseline)
    return max(ratios),metrics['rmse']['mean']


class DeadlineReached(BaseException):
    pass


class PromptTokenizer:
    def __init__(self,tokenizer,style):
        self.tokenizer=tokenizer
        self.style=style
    def encode(self,*args,**kwargs):
        return self.tokenizer.encode(*args,**kwargs)
    def apply_chat_template(self,messages,**kwargs):
        messages=copy.deepcopy(messages)
        messages[0]['content']=PROMPTS[self.style]+messages[0]['content']
        return self.tokenizer.apply_chat_template(messages,**kwargs)


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--deadline',required=True)
    args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline).timestamp()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=Path('docs/parameter-sweep-protocol.md').read_text()
    metadata=dict(protocol=protocol,protocol_sha256=hashlib.sha256(protocol.encode()).hexdigest(),deadline=args.deadline)
    write(args.output/'protocol.json',metadata)
    cases=json.loads(Path('bench/cases/synthetic.json').read_text())
    worlds=fresh_worlds([c['counts'] for c in cases])
    write(args.output/'fresh-cases.json',worlds)
    fresh_sha=hashlib.sha256((args.output/'fresh-cases.json').read_bytes()).hexdigest()
    # Selection uses only previously observed worlds, never fresh targets.
    dev_ids={c['id'] for c in cases if c['split']=='eval'}
    dev_ids=set(sorted(dev_ids)[:4])
    def screen(rows):
        return [(q,p) for q,p in rows if (q['split']=='calibration' and q['variant']=='base') or q['case_id'] in dev_ids]
    baseline=analyze(screen(load_rows(args.baseline)),dict(name='baseline-true-false-reused',style='baseline',labels=['TRUE','FALSE']))
    baseline['source_run']=str(args.baseline)
    leaderboard=[baseline]
    write(args.output/'leaderboard.json',leaderboard)
    from probabilistic_oracle.backends.vllm import VllmOracle
    from vllm import SamplingParams
    class BoundedOracle(VllmOracle):
        def score(self,request):
            if time.time()>=deadline:
                raise DeadlineReached()
            return super().score(request)
    oracle=BoundedOracle()
    tokenizer=oracle.tokenizer
    original_provenance=copy.deepcopy(oracle.provenance)
    def configure(config,sampling_temperature=0):
        oracle.tokenizer=PromptTokenizer(tokenizer,config['style'])
        oracle.sampling=dict(temperature=sampling_temperature,max_tokens=1,logprobs=-1,seed=0)
        oracle.params=SamplingParams(**oracle.sampling)
        oracle.provenance=copy.deepcopy(original_provenance)
        oracle.provenance.update(prompt_template='sweep-'+config['style'],prompt_prefix=PROMPTS[config['style']],
                                 sampling=oracle.sampling,verbalizers=config['labels'])
    try:
        from dataclasses import asdict
        query=synthetic_queries(cases[0])[0]
        check=[]
        for t in (0,.5,1):
            configure(baseline['config'],t)
            check.append(asdict(oracle.score(query.request)))
        write(args.output/'sampling-temperature-check.json',check)
        print('Sampling-temperature raw-logprob check complete',flush=True)
        for config in CONFIGS:
            remaining=deadline-time.time()
            if remaining<400:  # ~110 seconds screening + 240-second evaluation reserve.
                print(f"Stopping screen with {remaining:.0f}s reserved for evaluation",flush=True)
                break
            configure(config)
            queries=[q for c in cases for q in synthetic_queries(c,tuple(config['labels']))
                     if (q.split=='calibration' and q.variant=='base') or q.case_id in dev_ids]
            path=args.output/config['name']
            start=time.monotonic()
            run_benchmark(queries,oracle,path,dict(metadata,phase='development-screen'))
            try:
                result=analyze(load_rows(path),config)
            except ValueError as exc:
                write(path/'analysis-error.json',dict(error=str(exc)))
                print(f"{config['name']}: excluded due to errors: {exc}",flush=True)
                continue
            result['seconds']=time.monotonic()-start
            leaderboard.append(result)
            write(path/'summary.json',result)
            write(args.output/'leaderboard.json',leaderboard)
            print(json.dumps(dict(candidate=config['name'],calibration=result['calibration'],
                                  development_rmse=result['calibrated']['eval']['rmse']['mean'],
                                  gates=result['gates'],seconds=result['seconds'])),flush=True)
        best=min(leaderboard,key=rank)
        selection=dict(selected_utc=datetime.now(timezone.utc).isoformat(),config=best['config'],
                       calibration=best['calibration'],development_rank=rank(best),
                       development_gates=best['gates'],fresh_case_sha256=fresh_sha)
        write(args.output/'selection.json',selection)
        print('Frozen selection: '+json.dumps(selection),flush=True)
        configure(best['config'])
        queries=[q for c in worlds for q in synthetic_queries(c,tuple(best['config']['labels']))]
        queries += [q for q in load_queries(Path('bench/cases/synthetic.json'),Path('bench/cases/world.json'),tuple(best['config']['labels'])) if q.split=='world']
        path=args.output/'fresh-evaluation'
        run_benchmark(queries,oracle,path,dict(metadata,phase='fresh-evaluation',selection=selection))
        rows=load_rows(path)
        metrics=measurements(rows,best['calibration'])
        checks=gates(metrics)
        report=dict(selection=selection,raw=measurements(rows,fitted_constant=best['calibration']['fitted_constant']),calibrated=metrics,gates=checks,
                    decision='thresholds_met' if all(checks.values()) else 'thresholds_not_met',
                    total_queries=len(queries),screened_candidates=len(leaderboard),
                    completed_utc=datetime.now(timezone.utc).isoformat())
        write(args.output/'final-report.json',report)
        print('FINAL '+json.dumps(dict(decision=report['decision'],gates=checks)),flush=True)
    except DeadlineReached:
        write(args.output/'deadline.json',dict(status='time_limit',utc=datetime.now(timezone.utc).isoformat()))
        print('Time budget reached; partial observations retained without a final conclusion.',flush=True)


if __name__=='__main__':
    main()
