"""Matched numerical versus binary readout experiment on fresh finite worlds."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
import subprocess
import time

from probabilistic_oracle.oracle import normalize
from probabilistic_oracle.algebra import probability
from probabilistic_oracle.bench import synthetic_queries
from probabilistic_oracle.metrics import _aggregate, _case_metrics, interval, quality

FROZEN_CALIBRATION = dict(temperature=2, bias=-.5, fitted_constant=.47336894586894585)
LIMITS = dict(rmse=.1,negation_error=.1,conjunction_error=.1,bayes_error=.1,
              paraphrase_mae=.05,irrelevant_context_mae=.05)


def parse_probability(text, finish_reason):
    if finish_reason!='stop':
        raise ValueError(f'Incomplete numerical answer: finish_reason={finish_reason}')
    text=text.strip()
    if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',text):
        raise ValueError('Expected one numerical string without prose, percent, or fraction')
    return probability(float(text))


def fresh_cases(existing):
    rng=random.Random(2026092402)
    seen={tuple(c) for c in existing}
    cases=[]
    while len(cases)<8:
        cuts=[0]+sorted(rng.sample(range(1,100),3))+[100]
        counts=[b-a for a,b in zip(cuts,cuts[1:])]
        if tuple(counts) in seen:
            continue
        seen.add(tuple(counts))
        cases.append(dict(id=f'readout-{len(cases)+1:02}',split='eval',counts=counts))
    return cases


def calibrate_binary(p):
    p=min(1-1e-12,max(1e-12,probability(p)))
    z=(math.log(p)-math.log1p(-p))/FROZEN_CALIBRATION['temperature']+FROZEN_CALIBRATION['bias']
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))


def numerical_prompt(tokenizer, request):
    text=(
        'Assess the probability that the proposition is true given the context. '
        'Answer with exactly one number between 0 and 1. '
        'Do not include words, percentages, fractions, or an explanation.\n'
        f'Context: {request.context}\n'
    )
    if request.assumptions:
        text+='Assumptions (suppose these hold):\n'+'\n'.join(request.assumptions)+'\n'
    if request.observations:
        text+='Observations (new evidence):\n'+'\n'.join(request.observations)+'\n'
    text+=f'Proposition: {request.proposition}'
    return tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                                         add_generation_prompt=True,enable_thinking=False)


def summarize(queries, observations, backend=None):
    expected={q['id'] for q in queries}
    actual=[r['id'] for r in observations]
    if len(expected)!=len(queries) or len(actual)!=len(set(actual)) or expected!=set(actual):
        raise ValueError('Incomplete, duplicate, or unexpected query pairs')
    by_id={r['id']:r for r in observations}
    errors=[]
    valid={'numeric':set(),'binary':set()}
    groups={}
    for q in queries:
        groups.setdefault(q['case_id'],[]).append(q)
        for method in valid:
            record=by_id[q['id']][method]
            try:
                if record['status']!='ok':
                    raise ValueError(record.get('error','Failed observation'))
                if method=='numeric':
                    parsed=parse_probability(record['text'],record['finish_reason'])
                    if not math.isclose(parsed,probability(record['probability']),abs_tol=1e-12):
                        raise ValueError('Numerical probability does not match raw text')
                    if record['request']!=q['request']:
                        raise ValueError('Numerical request does not match query')
                else:
                    observation=record['observation']
                    raw,_=normalize(*observation['raw_logprobs'])
                    if not math.isclose(raw,probability(record['raw_score']),abs_tol=1e-12) or not math.isclose(raw,probability(observation['score']),abs_tol=1e-12):
                        raise ValueError('Binary score does not match raw log-probabilities')
                    if not math.isclose(calibrate_binary(raw),probability(record['calibrated_score']),abs_tol=1e-12):
                        raise ValueError('Binary calibration does not match frozen transform')
                    if observation['request']!=q['request']:
                        raise ValueError('Binary request does not match query')
                    if backend is not None and observation['provenance']!=backend:
                        raise ValueError('Binary provenance does not match manifest')
                valid[method].add(q['id'])
            except (KeyError,ValueError,TypeError) as exc:
                errors.append(dict(id=q['id'],method=method,error=str(exc)))
    complete=[case for case,qs in groups.items() if all(q['id'] in valid['numeric'] & valid['binary'] for q in qs)]
    report=dict(total_queries=len(queries),coverage={m+'_valid':len(ids) for m,ids in valid.items()},
                errors=errors,paired_cases=len(complete),excluded_cases=sorted(set(groups)-set(complete)))
    if not complete:
        return dict(report,decision='incomplete')
    rows_by_method={}
    for method,field in [('numeric','probability'),('binary_raw','raw_score'),('binary_calibrated','calibrated_score')]:
        source='numeric' if method=='numeric' else 'binary'
        rows_by_method[method]={case:[(q,by_id[q['id']][source][field]) for q in groups[case]] for case in complete}
        report[method]=_aggregate(rows_by_method[method],lambda p:p)
    baselines={}
    for name,constant in [('constant_half',.5),('calibration_mean',FROZEN_CALIBRATION['fitted_constant'])]:
        values=[quality([(constant,q['target']) for q in groups[case] if q['variant']=='base'])['rmse'] for case in complete]
        baselines[name]=dict(score=constant,rmse=interval(values))
    report['baselines']=baselines
    differences=[]
    for case in complete:
        a=_case_metrics(rows_by_method['numeric'][case],lambda p:p)['rmse']
        b=_case_metrics(rows_by_method['binary_calibrated'][case],lambda p:p)['rmse']
        differences.append(a-b)
    report['paired_rmse_difference']=interval(differences)
    for method in rows_by_method:
        scores=report[method]['eval']
        checks={k:scores[k] is not None and scores[k]['mean']<=limit for k,limit in LIMITS.items()}
        checks['beats_baselines']=all(scores['rmse']['mean']<v['rmse']['mean'] for v in baselines.values())
        checks['defined_bayes_paths']=scores['undefined_bayes_paths']==0
        if method=='binary_calibrated':
            checks['defined_bayes_paths'] &= report['binary_raw']['eval']['undefined_bayes_paths']==0
        report[method]['gates']=checks
    report['per_kind_valid_pairs']={}
    for kind in sorted({q['kind'] for q in queries}):
        qs=[q for q in queries if q['variant']=='base' and q['kind']==kind and q['id'] in valid['numeric'] & valid['binary']]
        per_kind={'count':len(qs)}
        if qs:
            for method,field in [('numeric','probability'),('binary_raw','raw_score'),('binary_calibrated','calibrated_score')]:
                source='numeric' if method=='numeric' else 'binary'
                per_kind[method]=quality([(by_id[q['id']][source][field],q['target']) for q in qs])
        report['per_kind_valid_pairs'][kind]=per_kind
    numeric_gates=report['numeric']['gates']
    numeric_accuracy=numeric_gates['rmse'] and numeric_gates['beats_baselines']
    binary_gates=report['binary_calibrated']['gates']
    binary_accuracy=binary_gates['rmse'] and binary_gates['beats_baselines']
    if errors:
        report['decision']='incomplete'
    elif len(complete)<8:
        report['decision']='insufficient_cases'
    elif numeric_accuracy and report['paired_rmse_difference']['high']<0:
        report['decision']='numeric_readout_supported'
    elif numeric_accuracy:
        report['decision']='numeric_accurate_difference_uncertain'
    elif report['paired_rmse_difference']['mean']<0:
        report['decision']='numeric_improves_but_fails'
    else:
        report['decision']='both_readouts_fail' if not binary_accuracy else 'binary_outperforms_numeric'
    return report


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def report_saved(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    digest=hashlib.sha256(json.dumps(manifest['queries'],sort_keys=True).encode()).hexdigest()
    if digest!=manifest['query_sha256']:
        raise ValueError('Query manifest hash mismatch')
    if hashlib.sha256(manifest['protocol'].encode()).hexdigest()!=manifest['protocol_sha256']:
        raise ValueError('Protocol hash mismatch')
    observations=[json.loads(line) for line in (directory/'pairs.jsonl').read_text().splitlines()]
    report=summarize(manifest['queries'],observations,manifest['backend'])
    report.update(query_sha256=manifest['query_sha256'],protocol_sha256=manifest['protocol_sha256'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous-sweep',type=Path)
    parser.add_argument('--deadline')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        report=report_saved(args.output)
        write(args.output/'report.json',report)
        print(json.dumps(report,indent=2))
        return
    if not args.previous_sweep or not args.deadline:
        parser.error('--previous-sweep and --deadline are required for inference')
    deadline=datetime.fromisoformat(args.deadline).timestamp()
    args.output.mkdir(parents=True,exist_ok=False)
    previous=json.loads((args.previous_sweep/'fresh-cases.json').read_text())
    original=json.loads(Path('bench/cases/synthetic.json').read_text())
    selection=json.loads((args.previous_sweep/'selection.json').read_text())
    if selection['config']!={'name':'baseline-a-b','style':'baseline','labels':['A','B']}:
        raise ValueError('Unexpected prior selected configuration')
    if any(selection['calibration'][k]!=v for k,v in FROZEN_CALIBRATION.items()):
        raise ValueError('Unexpected frozen calibration')
    cases=fresh_cases([c['counts'] for c in original+previous])
    queries=[q for c in cases for q in synthetic_queries(c,('A','B'))]
    encoded=[asdict(q) for q in queries]
    protocol=Path('docs/plans/2026-09-24-readout-comparison.md').read_text()
    manifest=dict(schema_version=1,started_utc=datetime.now(timezone.utc).isoformat(),cases=cases,queries=encoded,
                  query_sha256=hashlib.sha256(json.dumps(encoded,sort_keys=True).encode()).hexdigest(),
                  protocol=protocol,protocol_sha256=hashlib.sha256(protocol.encode()).hexdigest(),
                  previous_selection=selection,calibration=FROZEN_CALIBRATION,
                  code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  working_tree=subprocess.check_output(['git','status','--short'],text=True).strip())
    write(args.output/'manifest.json',manifest)
    from probabilistic_oracle.backends.vllm import VllmOracle
    from vllm import SamplingParams
    oracle=VllmOracle()
    numeric_settings=dict(temperature=0,max_tokens=16,seed=0)
    params=SamplingParams(**numeric_settings)
    manifest.update(backend=oracle.provenance,numeric_settings=numeric_settings,numeric_prompt_template='probability-number-v1')
    write(args.output/'manifest.json',manifest)
    def numeric(request):
        prompt=numerical_prompt(oracle.tokenizer,request)
        ids=oracle.tokenizer.encode(prompt,add_special_tokens=False)
        if len(ids)+numeric_settings['max_tokens']>oracle.engine['max_model_len']:
            raise ValueError('Numerical prompt exceeds context budget')
        start=time.perf_counter()
        output=oracle.llm.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)[0].outputs[0]
        result=dict(prompt=prompt,prompt_token_ids=ids,text=output.text,token_ids=list(output.token_ids),
                    finish_reason=output.finish_reason,seconds=time.perf_counter()-start,request=asdict(request))
        try:
            result.update(status='ok',probability=parse_probability(output.text,output.finish_reason))
        except ValueError as exc:
            result.update(status='error',error=str(exc))
        return result
    def binary(request):
        result=oracle.score(request)
        return dict(status='ok',raw_score=result.score,calibrated_score=calibrate_binary(result.score),observation=asdict(result))
    observations=[]
    with (args.output/'pairs.jsonl').open('x') as f:
        for i,q in enumerate(queries):
            pair={'id':q.id,'order':['numeric','binary'] if i%2==0 else ['binary','numeric']}
            for method in pair['order']:
                if time.time()>=deadline:
                    write(args.output/'deadline.json',dict(status='time_limit',completed_pairs=len(observations)))
                    return
                try:
                    pair[method]=(numeric if method=='numeric' else binary)(q.request)
                except Exception as exc:
                    pair[method]=dict(status='error',error=f'{type(exc).__name__}: {exc}')
            f.write(json.dumps(pair,allow_nan=False)+'\n');f.flush()
            observations.append(pair)
            if (i+1)%18==0:
                print(f'Completed {i+1}/{len(queries)} matched problems',flush=True)
    report=summarize(encoded,observations,manifest['backend'])
    report.update(query_sha256=manifest['query_sha256'],protocol_sha256=manifest['protocol_sha256'],
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/'report.json',report)
    print(json.dumps(dict(decision=report['decision'],coverage=report['coverage'],paired_cases=report['paired_cases'])),flush=True)


if __name__=='__main__':
    main()
