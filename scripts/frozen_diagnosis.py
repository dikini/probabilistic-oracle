"""Frozen Laya diagnosis experiment; exact finite-world expected scoring."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('laya_benchmark',ROOT/'scripts/laya_oracle.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
FAULTS=('coolant','bearing','sensor')
PATTERNS=('00','01','10','11')
PRIOR=[.5,.3,.2]
X=[.8,.4,.1];Y=[.7,.3,.2]
DESCRIPTIONS={'00':'Temperature is normal and vibration is absent.',
              '01':'Temperature is normal and vibration is present.',
              '10':'Temperature is high and vibration is absent.',
              '11':'Temperature is high and vibration is present.'}


def worlds():
    result={}
    for name in ('independent','correlated'):
        joint={}
        for i,h in enumerate(FAULTS):
            both=X[i]*Y[i] if name=='independent' else [.68,.28,.09][i]
            joint[h]=[1-X[i]-Y[i]+both,Y[i]-both,X[i]-both,both]
        result[name]=dict(prior=list(PRIOR),joint=joint)
    return result


def marginals(joint):
    return [joint[2]+joint[3],joint[1]+joint[3]]


def normalize(xs):
    if any(not math.isfinite(v) or v<0 for v in xs) or sum(xs)<=0:raise ValueError('Invalid probability weights')
    total=sum(xs);return [v/total for v in xs]


def update(prior,likelihoods):
    if len(prior)!=3 or len(likelihoods)!=3:raise ValueError('Expected three faults')
    return normalize([p*l for p,l in zip(prior,likelihoods)])


def truth(world,pattern):
    return update(world['prior'],[world['joint'][h][PATTERNS.index(pattern)] for h in FAULTS])


def evidence_mass(world,pattern):
    return sum(world['prior'][i]*world['joint'][h][PATTERNS.index(pattern)] for i,h in enumerate(FAULTS))


def naive(world,pattern):
    likes=[]
    for h in FAULTS:
        x,y=marginals(world['joint'][h]);likes.append((x if pattern[0]=='1' else 1-x)*(y if pattern[1]=='1' else 1-y))
    return update(world['prior'],likes)


def table(world,coverage):
    text='Fictional machine population. Exactly one fault exists: coolant, bearing, or sensor. X means high temperature; Y means vibration. These are population probabilities, not observations of this machine.\n'
    for i,h in enumerate(FAULTS):
        x,y=marginals(world['joint'][h])
        text+=f'{h}: prior={world["prior"][i]:.2f}; P(X present|{h})={x:.2f}; P(Y present|{h})={y:.2f}.\n'
    if coverage=='full':
        text+='Joint symptom probabilities conditional on each fault (00=normal temperature/no vibration; 01=normal temperature/vibration; 10=high temperature/no vibration; 11=high temperature/vibration):\n'
        for h in FAULTS:text+=h+': '+', '.join(f'{p}={v:.2f}' for p,v in zip(PATTERNS,world['joint'][h]))+'.\n'
    else:text+='Joint symptom probabilities are not supplied.\n'
    return text


def observation(pattern,variant):
    x,y=pattern
    if variant==0:return DESCRIPTIONS[pattern]
    if variant==1:return f'The machine {"vibrates" if y=="1" else "does not vibrate"}. Its temperature is {"high" if x=="1" else "normal"}.'
    return f'Inspection found {"elevated" if x=="1" else "normal"} temperature, with {"vibration detected" if y=="1" else "no vibration detected"}.'


def ordered(options,order):
    return dict(reversed(list(options.items()))) if order else dict(options)


def make_requests():
    rows=[]
    for v,order in itertools.product(range(3),range(2)):
        for pattern in PATTERNS:
            instruction=['Which symptom combination does the report explicitly describe?',
                         'Identify the temperature and vibration state stated in the report.',
                         'Select the observed pair of symptoms, using only the report.'][v]
            opts=ordered(DESCRIPTIONS,order)
            rows.append(dict(id=f'extract/{pattern}/{v}/{order}',role='extract',pattern=pattern,variant=v,order=order,
                state=observation(pattern,v),question=dict(type='choice',instructions=instruction,criteria=opts),semantics={k:k for k in opts}))
        for name,world in worlds().items():
            for coverage in ('full','reduced'):
                prefix=f'{name}/{coverage}/{v}/{order}';state=table(world,coverage)
                def add(role,suffix,instruction,options,request_state=state,**extra):
                    opts=ordered(options,order)
                    rows.append(dict(id=f'{role}/{prefix}{suffix}',role=role,world=name,coverage=coverage,variant=v,order=order,
                        state=request_state,question=dict(type='choice',instructions=instruction,criteria=opts),semantics={k:k for k in opts},**extra))
                options={h:f'The fault is {h}.' for h in FAULTS}
                instruction=['For a random machine before observing symptoms, which fault does it have?',
                    'An unobserved machine is sampled from this population. Identify its fault.',
                    'Before any case evidence, select the fault of a uniformly sampled machine.'][v]
                add('prior','',instruction,options)
                for h in FAULTS:
                    if coverage=='full':
                        instruction=[f'For a random machine known to have the {h} fault, which symptom combination occurs?',
                            f'Given fault {h}, identify the symptom pair of a randomly sampled machine.',
                            f'Assume the machine has fault {h}. Select its temperature and vibration combination before observing it.'][v]
                        add('joint','/'+h,instruction,DESCRIPTIONS,fault=h)
                    for axis,symptom in [('x','high temperature'),('y','vibration')]:
                        instruction=[f'For a random machine with fault {h}, is {symptom} present or absent?',
                            f'Given the {h} fault, select whether a sampled machine has {symptom}.',
                            f'Assume fault {h}. Classify the unobserved machine by presence of {symptom}.'][v]
                        add(axis,'/'+h,instruction,{'present':f'{symptom} is present.','absent':f'{symptom} is absent.'},fault=h)
                for pattern in PATTERNS:
                    instruction=['Given the table and observed report, which fault does this machine have?',
                        'Use the supplied probabilities and symptoms to diagnose the fault.',
                        'Select the fault for this machine after considering the population table and its report.'][v]
                    add('direct','/'+pattern,instruction,options,state+'\nObserved report: '+observation(pattern,v),pattern=pattern)
    return rows


def scores(pred,target):
    base.distribution(pred,3);base.distribution(target,3)
    return dict(accuracy=target[max(range(3),key=lambda i:pred[i])],
                log_loss=-sum(t*math.log(max(p,1e-12)) for p,t in zip(pred,target)),
                brier=sum(p*p for p in pred)-2*sum(p*t for p,t in zip(pred,target))+1,
                zero_probabilities=sum(p==0 for p in pred))


def report_saved(directory):
    manifest=json.loads((directory/'manifest.json').read_text());requests=manifest['requests']
    if base.digest(requests)!=manifest['requests_sha256']:raise ValueError('Request hash mismatch')
    records=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    if len(records)!=len(requests) or {r['id'] for r in records}!={r['id'] for r in requests}:raise ValueError('Incomplete or duplicate records')
    by={r['id']:r for r in records};probabilities={};seconds={}
    for q in requests:
        r=by[q['id']]
        if r['request']!=q or r['status']!='ok':raise ValueError('Failed or mismatched request')
        probabilities[q['id']]=base.checked_probabilities(q,r);seconds[q['id']]=r['seconds']
    results=dict(total=len(records),status='complete',readouts={})
    for mode,index in [('raw',0),('shipped',1)]:
        probs={key:value[index] for key,value in probabilities.items()}
        predictions=[];extraction=[];parameter_errors=defaultdict(list)
        for name,world in worlds().items():
            for coverage,v,order in itertools.product(('full','reduced'),range(3),range(2)):
                prefix=f'{name}/{coverage}/{v}/{order}';pid='prior/'+prefix;prior=[probs[pid][h] for h in FAULTS]
                parameter_errors[f'{name}/{coverage}/prior'].extend(abs(a-b) for a,b in zip(prior,world['prior']))
                for h in FAULTS:
                    for axis,k in [('x',0),('y',1)]:parameter_errors[f'{name}/{coverage}/marginal'].append(abs(probs[f'{axis}/{prefix}/{h}']['present']-marginals(world['joint'][h])[k]))
                    if coverage=='full':parameter_errors[f'{name}/{coverage}/joint'].extend(abs(probs[f'joint/{prefix}/{h}'][p]-t) for p,t in zip(PATTERNS,world['joint'][h]))
                for pattern in PATTERNS:
                    target=truth(world,pattern);weight=evidence_mass(world,pattern)/6
                    eid=f'extract/{pattern}/{v}/{order}';observed=max(probs[eid],key=probs[eid].get)
                    did=f'direct/{prefix}/{pattern}';extraction.append(dict(world=name,coverage=coverage,pattern=pattern,variant=v,order=order,correct=observed==pattern,weight=weight))
                    def add(method,pred,used):
                        predictions.append(dict(world=name,coverage=coverage,variant=v,order=order,pattern=pattern,observed=observed,
                            method=method,prediction=pred,target=target,weight=weight,request_ids=used,metrics=scores(pred,target)))
                    add('direct',[probs[did][h] for h in FAULTS],[did])
                    add('prior_only',world['prior'],[])
                    add('reference_joint',target,[])
                    add('reference_naive',naive(world,pattern),[])
                    add('hybrid_naive',naive(world,observed),[eid])
                    if coverage=='full':add('hybrid_joint',truth(world,observed),[eid])
                    for evidence,label in [(observed,''),(pattern,'_oracle_evidence')]:
                        used=[pid]+[f'{axis}/{prefix}/{h}' for h in FAULTS for axis in ('x','y')]
                        likes=[]
                        for h in FAULTS:
                            x=probs[f'x/{prefix}/{h}'][('present' if evidence[0]=='1' else 'absent')]
                            y=probs[f'y/{prefix}/{h}'][('present' if evidence[1]=='1' else 'absent')]
                            likes.append(x*y)
                        add('factorized_naive'+label,update(prior,likes),used+([eid] if not label else []))
                        if coverage=='full':
                            used=[pid]+[f'joint/{prefix}/{h}' for h in FAULTS]
                            add('factorized_joint'+label,update(prior,[probs[f'joint/{prefix}/{h}'][evidence] for h in FAULTS]),used+([eid] if not label else []))
        summaries={}
        for name,coverage in itertools.product(worlds(),('full','reduced')):
            methods=sorted({r['method'] for r in predictions if r['world']==name and r['coverage']==coverage})
            for method in methods:
                group=[r for r in predictions if r['world']==name and r['coverage']==coverage and r['method']==method]
                if abs(sum(r['weight'] for r in group)-1)>1e-10:raise ValueError('Unnormalized evaluation weights')
                summary={metric:sum(r['weight']*r['metrics'][metric] for r in group) for metric in ('accuracy','log_loss','brier')}
                by_pattern=defaultdict(list)
                for r in group:by_pattern[r['pattern']].append(r['prediction'])
                summary['max_presentation_tv']=max(.5*sum(abs(x-y) for x,y in zip(a,b)) for vals in by_pattern.values() for a,b in itertools.combinations(vals,2))
                used=sorted({i for r in group for i in r['request_ids']});params=[i for i in used if i.split('/')[0] in ('prior','joint','x','y')]
                summary.update(unique_requests=len(used),parameter_requests=len(params),request_seconds=sum(seconds[i] for i in used),
                    parameter_seconds=sum(seconds[i] for i in params),diagnosis_presentations=len(group),zero_probabilities=sum(r['metrics']['zero_probabilities'] for r in group))
                summaries[f'{name}/{coverage}/{method}']=summary
        results['readouts'][mode]=dict(summary=summaries,predictions=predictions,
            parameter_mean_absolute_error={k:sum(v)/len(v) for k,v in parameter_errors.items()},
            extraction=dict(unique_requests=24,correct=sum(max(probs[q['id']],key=probs[q['id']].get)==q['pattern'] for q in requests if q['role']=='extract'),
                weighted_accuracy={f'{name}/{coverage}':sum(r['weight']*r['correct'] for r in extraction if r['world']==name and r['coverage']==coverage) for name,coverage in itertools.product(worlds(),('full','reduced'))}))
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--deadline');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.report_only:base.write(args.output/'report.json',report_saved(args.output));return
    if not args.deadline:parser.error('--deadline required')
    requests=make_requests();args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(requests=requests,requests_sha256=base.digest(requests),worlds=worlds(),model=base.MODEL,revision=base.REVISION,
        protocol=(ROOT/'docs/plans/2026-09-28-frozen-diagnosis.md').read_text(),
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree=subprocess.check_output(['git','status','--short'],text=True).strip(),started_utc=datetime.now(timezone.utc).isoformat())
    base.write(args.output/'manifest.json',manifest)
    import numpy as np
    import torch
    from importlib.metadata import version
    from laya.agent import Agent
    from laya.common import build_sequence,render_options,QTYPES,temp_bucket
    if version('laya')!='0.3.21':raise ValueError('Unexpected SDK version')
    torch.manual_seed(0);np.random.seed(0)
    class RecordingAgent(Agent):
        def _forward(self,b):
            logits,act=super()._forward(b);self.observed_logits=logits.tolist();return logits,act
    agent=RecordingAgent(base.MODEL,revision=base.REVISION,device='cuda',fast=False,compile=False)
    manifest.update(versions={p:version(p) for p in ('laya','torch','transformers','huggingface-hub','numpy','safetensors')},config=agent.cfg,
                    temperature=agent.temperature,temperature_by_options=agent.temperature_by_options,device=str(agent.device),dtype=str(agent.dtype),amp_enabled=agent.amp_enabled)
    base.write(args.output/'manifest.json',manifest)
    deadline=datetime.fromisoformat(args.deadline).timestamp();schedule=list(requests);random.Random(2026092802).shuffle(schedule)
    with (args.output/'records.jsonl').open('x') as stream:
        for i,q in enumerate(schedule):
            if time.time()>=deadline:raise TimeoutError('Inference deadline')
            r=dict(id=q['id'],request=q)
            try:
                internal=agent._to_internal(q['question']);seq,markers,stats=build_sequence(agent.tok,q['state'],internal,512,192,return_stats=True)
                full,_,_=build_sequence(agent.tok,q['state'],internal,4096,2048,return_stats=True)
                if seq!=full or any(len(agent.tok.encode(' '+o,add_special_tokens=False))>48 for o in render_options(internal)):raise ValueError('Input would be truncated')
                start=time.perf_counter();response=agent.predict(q['state'],{'answer':q['question']});elapsed=time.perf_counter()-start
                qt=QTYPES[internal['t']];temperature=agent.temperature_by_options.get(temp_bucket(qt,len(markers)),agent.temperature[qt])
                r.update(status='ok',response=response,logits=agent.observed_logits[0],shipped_temperature=temperature,token_ids=seq,markers=markers,option_stats=stats,seconds=elapsed,device=str(agent.device))
                base.checked_probabilities(q,r)
            except Exception as exc:r.update(status='error',error=f'{type(exc).__name__}: {exc}')
            stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            if (i+1)%20==0:print(f'Completed {i+1}/{len(requests)}',flush=True)
    base.write(args.output/'report.json',report_saved(args.output))


if __name__=='__main__':main()
