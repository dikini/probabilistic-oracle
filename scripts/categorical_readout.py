"""Prospective categorical truth-scale experiment; raw observations stay separate."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import subprocess
import time

from probabilistic_oracle.bench import synthetic_queries
from probabilistic_oracle.metrics import TEMPERATURES, quality, interval, _aggregate, _case_metrics
from probabilistic_oracle.oracle import prepare_prompt

CATEGORIES={2:['TRUE','FALSE'],3:['TRUE','NEITHER TRUE NOR FALSE','FALSE'],
            5:['TRUE','MOSTLY TRUE','NEITHER TRUE NOR FALSE','MOSTLY FALSE','FALSE']}
BIASES=(-2,-1,-.5,0,.5,1,2)
LIMITS=dict(rmse=.1,negation_error=.1,conjunction_error=.1,bayes_error=.1,paraphrase_mae=.05,irrelevant_context_mae=.05)
PROTOCOL=Path('docs/plans/2026-09-24-categorical-readout.md')


def anchors(n):
    if n not in CATEGORIES: raise ValueError('Unsupported scale')
    return [1-i/(n-1) for i in range(n)]


def score_logs(logs,values,temperature=1,bias=0):
    if len(logs)!=len(values) or len(logs)<2: raise ValueError('Mismatched candidates')
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x>0 for x in logs):
        raise ValueError('Invalid log-probability')
    if any(not math.isfinite(x) or not 0<=x<=1 for x in values): raise ValueError('Invalid anchor')
    if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(bias): raise ValueError('Invalid calibration')
    mass=sum(math.exp(x) for x in logs)
    if mass>1+1e-5: raise ValueError('Candidate mass exceeds one')
    z=[x/temperature+bias*v for x,v in zip(logs,values)]
    if any(not math.isfinite(x) for x in z): raise ValueError('Calibration overflow')
    weights=[math.exp(x-max(z)) for x in z];total=sum(weights)
    weights=[x/total for x in weights]
    return sum(w*v for w,v in zip(weights,values)),mass,weights


def extract(logs,ids,values):
    if len(ids)!=len(set(ids)) or any(i not in logs for i in ids):
        raise ValueError('Missing or duplicate candidate token')
    return score_logs([logs[i] for i in ids],values)


def prompt_text(request,n):
    text='Assess the proposition given the context. Choose exactly one truth category.\n'
    text+='\n'.join(f'{chr(65+i)} = {category}' for i,category in enumerate(CATEGORIES[n]))
    text+='\nAnswer with exactly one category letter and no explanation.\n'
    text+=f'Context: {request.context}\n'
    if request.assumptions: text+='Assumptions (suppose these hold):\n'+'\n'.join(request.assumptions)+'\n'
    if request.observations: text+='Observations (new evidence):\n'+'\n'.join(request.observations)+'\n'
    return text+f'Proposition: {request.proposition}'


def prepare(tokenizer,request,n):
    if n==2: return prepare_prompt(tokenizer,request)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':prompt_text(request,n)}],tokenize=False,
                                         add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer.encode(prompt,add_special_tokens=False);candidates=[]
    for label in [chr(65+i) for i in range(n)]:
        token=tokenizer.encode(label,add_special_tokens=False)
        if len(token)!=1 or tokenizer.encode(prompt+label,add_special_tokens=False)!=ids+token:
            raise ValueError('Invalid candidate token boundary')
        candidates+=token
    if len(set(candidates))!=n: raise ValueError('Duplicate candidate IDs')
    return prompt,ids,candidates


def fresh_cases(existing):
    rng=random.Random(2026092403);seen={tuple(c) for c in existing};cases=[]
    while len(cases)<12:
        cuts=[0]+sorted(rng.sample(range(1,100),3))+[100]
        counts=[b-a for a,b in zip(cuts,cuts[1:])]
        if tuple(counts) in seen: continue
        seen.add(tuple(counts));i=len(cases)
        cases.append(dict(id=f'categorical-{i+1:02}',split='calibration' if i<4 else 'eval',counts=counts))
    return cases


def fit(rows,values):
    selected=[(logs,q['target']) for q,logs in rows if q['split']=='calibration' and q['variant']=='base']
    if not selected: raise ValueError('No calibration data')
    def objective(t,b): return quality([(score_logs(logs,values,t,b)[0],y) for logs,y in selected])['log_loss']
    t,b=min(((t,b) for t in TEMPERATURES for b in BIASES),key=lambda tb:(objective(*tb),abs(tb[0]-1),abs(tb[1]),tb))
    return dict(temperature=t,bias=b,log_loss=objective(t,b),query_count=len(selected))


def digest(data): return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def write(path,data): path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def validate_records(manifest,records):
    expected=[(q['id'],n) for q in manifest['queries'] for n in CATEGORIES]
    actual=[(r['id'],r['scale']) for r in records]
    if len(expected)!=len(set(expected)) or len(actual)!=len(set(actual)) or set(expected)!=set(actual):
        raise ValueError('Incomplete, duplicate, or unexpected observations')
    queries={q['id']:q for q in manifest['queries']}
    for r in records:
        if r['status']!='ok': raise ValueError(f"Failed observation: {r['id']} scale {r['scale']}: {r.get('error')}")
        n=r['scale'];ids=r['candidate_token_ids']
        p,mass,_=extract(dict(zip(ids,r['raw_logprobs'])),ids,anchors(n))
        if len(ids)!=n or len(r['raw_logprobs'])!=n: raise ValueError('Wrong candidate count')
        if r['request']!=queries[r['id']]['request'] or r['provenance']!=manifest['backend']:
            raise ValueError('Request or provenance mismatch')
        if r['categories']!=CATEGORIES[n] or r['anchors']!=anchors(n): raise ValueError('Scale mismatch')
        if not math.isclose(p,r['raw_score'],abs_tol=1e-12) or not math.isclose(mass,r['candidate_mass'],abs_tol=1e-12):
            raise ValueError('Stored score or mass does not match raw logs')
    return {(r['id'],r['scale']):r for r in records}


def summarize(manifest,records,frozen):
    if digest(manifest['queries'])!=manifest['query_sha256'] or digest(manifest['protocol'])!=manifest['protocol_sha256']:
        raise ValueError('Manifest hash mismatch')
    by=validate_records(manifest,records);qs=manifest['queries']
    cal=[q for q in qs if q['split']=='calibration' and q['variant']=='base']
    evaluation=[q for q in qs if q['split']=='eval']
    if len({q['case_id'] for q in cal})<4 or len({q['case_id'] for q in evaluation})<8:
        raise ValueError('Insufficient worlds')
    if {q['case_id'] for q in cal}&{q['case_id'] for q in evaluation}: raise ValueError('Split leakage')
    report=dict(total_requests=len(records),decision='thresholds_not_met',methods={},baselines={},paired_differences={})
    for name,c in [('constant_half',.5),('calibration_mean',mean(q['target'] for q in cal))]:
        vals=[quality([(c,q['target']) for q in evaluation if q['case_id']==case and q['variant']=='base'])['rmse']
              for case in sorted({q['case_id'] for q in evaluation})]
        report['baselines'][name]=dict(score=c,rmse=interval(vals))
    all_groups={}
    for n in CATEGORIES:
        rows=[(q,by[q['id'],n]['raw_logprobs']) for q in qs]
        fitted=fit(rows,anchors(n))
        if fitted!=frozen[str(n)]: raise ValueError('Calibration differs from frozen parameters')
        result=dict(calibration=fitted)
        for mode,t,b in [('raw',1,0),('calibrated',fitted['temperature'],fitted['bias'])]:
            groups={}
            for q,logs in rows:
                if q['split']=='eval':groups.setdefault(q['case_id'],[]).append((q,score_logs(logs,anchors(n),t,b)[0]))
            result[mode]=_aggregate(groups,lambda p:p)['eval'];all_groups[n,mode]=groups
        ev=[by[q['id'],n] for q in evaluation]
        result['candidate_mass']=dict(mean=mean(r['candidate_mass'] for r in ev),minimum=min(r['candidate_mass'] for r in ev))
        result['raw_argmax_counts']=dict(Counter(CATEGORIES[n][max(range(n),key=lambda i:r['raw_logprobs'][i])] for r in ev))
        result['generated_in_candidates']=sum(bool(r['generated_token_ids']) and r['generated_token_ids'][0] in r['candidate_token_ids'] for r in ev)
        scores=result['calibrated']
        gates={k:scores[k] is not None and scores[k]['mean']<=v for k,v in LIMITS.items()}
        gates['beats_baselines']=all(scores['rmse']['mean']<v['rmse']['mean'] for v in report['baselines'].values())
        gates['defined_bayes_paths']=scores['undefined_bayes_paths']==0 and result['raw']['undefined_bayes_paths']==0
        result['gates']=gates;result['decision']='thresholds_met' if all(gates.values()) else 'thresholds_not_met'
        report['methods'][str(n)]=result
    for n in (3,5):
        a=all_groups[n,'calibrated'];b=all_groups[2,'calibrated']
        report['paired_differences'][str(n)]=interval([_case_metrics(a[c],lambda p:p)['rmse']-_case_metrics(b[c],lambda p:p)['rmse'] for c in a])
    if any(report['methods'][str(n)]['decision']=='thresholds_met' for n in (3,5)):report['decision']='categorical_thresholds_met'
    return report


def report_saved(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    records=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    frozen=json.loads((directory/'calibration.json').read_text())
    return summarize(manifest,records,frozen)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--history',type=Path)
    parser.add_argument('--deadline');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.report_only:
        report=report_saved(args.output);write(args.output/'report.json',report);print(json.dumps(report));return
    if not args.history or not args.deadline:parser.error('--history and --deadline required')
    deadline=datetime.fromisoformat(args.deadline).timestamp()
    existing=json.loads(Path('bench/cases/synthetic.json').read_text())
    existing+=json.loads((args.history/'parameter-sweep-20260924/fresh-cases.json').read_text())
    existing+=json.loads((args.history/'readout-comparison-20260924/manifest.json').read_text())['cases']
    cases=fresh_cases([c['counts'] for c in existing])
    queries=[q for c in cases for q in synthetic_queries(c,('A','B')) if q.split=='eval' or q.variant=='base']
    encoded=json.loads(json.dumps([asdict(q) for q in queries]));protocol=PROTOCOL.read_text()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(cases=cases,queries=encoded,query_sha256=digest(encoded),protocol=protocol,protocol_sha256=digest(protocol),
                  code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  working_tree=subprocess.check_output(['git','status','--short'],text=True).strip(),
                  started_utc=datetime.now(timezone.utc).isoformat(),excluded_counts=[c['counts'] for c in existing],
                  calibration_grid=dict(temperatures=TEMPERATURES,biases=BIASES))
    write(args.output/'manifest.json',manifest)
    from probabilistic_oracle.backends.vllm import VllmOracle
    oracle=VllmOracle();manifest['backend']=dict(oracle.provenance,prompt_template='categorical-v1-with-binary-v1-control')
    write(args.output/'manifest.json',manifest)
    records=[];frozen=None
    with (args.output/'records.jsonl').open('x') as f:
        for i,q in enumerate(queries):
            if q.split=='eval' and frozen is None:
                frozen={str(n):fit([(encoded[j],r['raw_logprobs']) for j,cq in enumerate(queries[:i])
                                   for r in records if r['id']==cq.id and r['scale']==n and r['status']=='ok'],anchors(n)) for n in CATEGORIES}
                if any(v['query_count']!=24 for v in frozen.values()):raise ValueError('Incomplete calibration data')
                write(args.output/'calibration.json',frozen)
                print('Calibration frozen; starting untouched evaluation',flush=True)
            scales=list(CATEGORIES);scales=scales[i%3:]+scales[:i%3]
            for n in scales:
                if time.time()>=deadline:raise TimeoutError('Experiment deadline reached; raw records retained')
                r=dict(id=q.id,scale=n)
                try:
                    prompt,ids,candidates=prepare(oracle.tokenizer,q.request,n)
                    if len(ids)+1>oracle.engine['max_model_len']:raise ValueError('Prompt exceeds context budget')
                    start=time.perf_counter()
                    out=oracle.llm.generate([{'prompt_token_ids':ids}],oracle.params,use_tqdm=False)[0].outputs[0]
                    if not out.logprobs:raise ValueError('No token log-probabilities')
                    logs={t:out.logprobs[0][t].logprob for t in candidates if t in out.logprobs[0]}
                    p,mass,weights=extract(logs,candidates,anchors(n))
                    r.update(status='ok',request=asdict(q.request),prompt=prompt,prompt_token_ids=ids,candidate_token_ids=candidates,
                             raw_logprobs=[logs[t] for t in candidates],raw_score=p,candidate_mass=mass,raw_weights=weights,
                             generated_token_ids=list(out.token_ids),text=out.text,finish_reason=out.finish_reason,
                             categories=CATEGORIES[n],anchors=anchors(n),seconds=time.perf_counter()-start,provenance=manifest['backend'])
                except Exception as exc:r.update(status='error',error=f'{type(exc).__name__}: {exc}')
                f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();records.append(r)
            if (i+1)%6==0:print(f'Completed {i+1}/{len(queries)} matched queries ({len(records)} requests)',flush=True)
    report=report_saved(args.output);write(args.output/'report.json',report)
    print(json.dumps(dict(decision=report['decision'],total_requests=report['total_requests'])),flush=True)


if __name__=='__main__':main()
