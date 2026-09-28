"""Fixed native-Laya knowledge, grounding and Bayesian-prior diagnostic."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
MODEL='convaiinnovations/laya'
REVISION='55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851'


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()


def write(path,x):
    path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def make_requests():
    import importlib.util
    spec=importlib.util.spec_from_file_location('seasonal',ROOT/'scripts/seasonal_knowledge.py')
    seasonal=importlib.util.module_from_spec(spec);spec.loader.exec_module(seasonal)
    facts=json.loads((ROOT/'bench/cases/seasonal-facts.json').read_text())
    choices={k:k for k in ('cold','mild','hot','unknown')}
    rows=[]
    for supplied in (False,True):
        for q in seasonal.queries(seasonal.load_cases(),facts if supplied else None):
            rows.append(dict(id=f"weather/{int(supplied)}/{q['id']}",suite='weather',supplied=supplied,
                case=q['case'],variant=q['variant'],target=facts[q['case']['country']]['provided_labels'][q['case']['month']],
                state=q['text'],question=dict(type='choice',instructions='Answer the temperature-category question in the state.',criteria=choices),semantics=choices))
    for c in seasonal.load_cases():
        for label in ('cold','hot'):
            for reverse in (False,True):
                opts=dict(list(choices.items())[::(-1 if reverse else 1)])
                rows.append(dict(id=f"changed/{c['id']}/{label}/{int(reverse)}",suite='changed',target=label,
                    state=f"This fictional reference profile overrides real-world climate knowledge for this task. For {c['country']} in {c['month']}, the typical daytime temperature category is {label}.",
                    question=dict(type='choice',instructions=f"According to this fictional profile, which category applies to {c['country']} in {c['month']}?",criteria=opts),semantics=opts))
    for split,counts in [('calibration',[10,20,30,40,60,70,80,90]),('test',[5,15,25,35,45,55,65,75,85,95])]:
        for i,count in enumerate(counts):
            if split=='calibration':
                state=f"Fictional inventory lot K{i} contains exactly 100 sealed packages: {count} marked amber and {100-count} marked violet. One package is drawn uniformly at random. Its mark has not been observed. There is no other information."
                ins='Is the drawn package marked amber?';yes='The drawn package is marked amber.';no='The drawn package is marked violet.'
            else:
                state=f"Fictional support archive R{i} has 100 records. Exactly {count} concern billing; the other {100-count} concern delivery. A record is selected uniformly at random, but its contents remain hidden. No additional evidence is available."
                ins='Does the selected record concern billing?';yes='The selected record concerns billing.';no='The selected record concerns delivery.'
            for swap in (False,True):
                for reverse in (False,True):
                    pos,neg=('B','A') if swap else ('A','B');criteria={pos:yes,neg:no}
                    if reverse:criteria=dict(reversed(list(criteria.items())))
                    rows.append(dict(id=f'rate/{split}/{i}/{int(swap)}{int(reverse)}',suite='rate',split=split,case_id=f'{split}/{i}',target=count/100,state=state,
                        question=dict(type='choice',instructions=ins,criteria=criteria),semantics={pos:'positive',neg:'negative'}))
            rows.append(dict(id=f'rate/{split}/{i}/noul',suite='rate',split=split,case_id=f'{split}/{i}',target=count/100,state=state,
                question=dict(type='noul',instructions=ins),semantics={'false':'negative','true':'positive'}))
    return rows



def make_followup_requests():
    import copy
    rows=[]
    facts=json.loads((ROOT/'bench/cases/seasonal-facts.json').read_text())
    for original in make_requests():
        if original['suite']!='weather':continue
        q=copy.deepcopy(original);q['id']='native/'+q['id']
        text=q['state']
        if q['supplied']:
            block='Background facts for this question (take as given):\n'+facts[q['case']['country']]['text']
            assert text.startswith(block+'\n\n')
            q['state']=block;text=text[len(block)+2:]
        else:q['state']='Use general world knowledge.'
        q['question']['instructions']=text;rows.append(q)
    billing=[
        'I was charged twice for the same purchase. Please refund the duplicate.',
        'The invoice total is higher than the agreed price.',
        'Please update the payment card used for my subscription.',
        'My refund has not appeared in my bank account.',
        'I need a receipt for the amount you debited yesterday.',
        'The discount code was accepted but the full price was charged.',
        'Why is there an unexpected fee on my monthly bill?',
        'My payment was declined despite sufficient funds.',
        'Please correct the tax amount on this invoice.',
        'You billed me after I cancelled the service.',
    ]
    delivery=[
        'The parcel tracking page has not updated for a week.',
        'My package was sent to the wrong street address.',
        'The courier marked my order delivered, but it never arrived.',
        'Can you change the destination before the parcel is dispatched?',
        'When will the replacement package reach my house?',
        'The delivery driver could not find our building.',
        'Please leave the shipment at the collection point.',
        'The parcel is stuck at the regional sorting depot.',
        'I need to reschedule the courier visit for tomorrow.',
        'The shipment arrived three days later than promised.',
    ]
    for cohort,count in [('low',4),('balanced',10),('high',16)]:
        labels=['billing']*count+['delivery']*(20-count)
        random.Random(20260928+count).shuffle(labels)
        for i,label in enumerate(labels):
            text=(billing if label=='billing' else delivery)[i%10]
            for order in (0,1):
                criteria={'billing':'Payments, charges, invoices, refunds or account billing.', 'delivery':'Shipping, parcels, couriers, tracking or delivery destinations.'}
                if order:criteria=dict(reversed(list(criteria.items())))
                rows.append(dict(id=f'corpus/{cohort}/{i}/{order}',suite='corpus',cohort=cohort,case_id=f'{cohort}/{i}',order=order,target=label,
                    state=f'Record {cohort}-{i}: '+text,question=dict(type='choice',instructions='Which department should handle this request?',criteria=criteria),semantics={k:k for k in criteria}))
    return rows


def distribution(values,k):
    if len(values)!=k or any(not math.isfinite(v) or v<0 or v>1 for v in values) or abs(sum(values)-1)>1e-6:
        raise ValueError('Invalid or missing probability distribution')
    return list(values)


def softmax(logits,temperature=1):
    if not logits or not all(math.isfinite(v) for v in logits) or temperature<=0:raise ValueError('Invalid logits/temperature')
    z=[v/temperature for v in logits];maximum=max(z);exps=[math.exp(v-maximum) for v in z];total=sum(exps)
    return distribution([v/total for v in exps],len(logits))


def positive_probability(request,probabilities):
    if set(probabilities)!=set(request['semantics']):raise ValueError('Missing or unexpected options')
    distribution(list(probabilities.values()),len(request['semantics']))
    return sum(p for key,p in probabilities.items() if request['semantics'][key]=='positive')


def posterior(p):
    return .8*p/(.8*p+.2*(1-p))


def metrics(ps,targets):
    if not ps or len(ps)!=len(targets):raise ValueError('Unpaired metrics')
    eps=1e-12
    return dict(rmse=math.sqrt(sum((p-y)**2 for p,y in zip(ps,targets))/len(ps)),
        expected_brier=sum(y*(1-p)**2+(1-y)*p*p for p,y in zip(ps,targets))/len(ps),
        expected_log_loss=-sum(y*math.log(max(eps,p))+(1-y)*math.log(max(eps,1-p)) for p,y in zip(ps,targets))/len(ps))


def scale(p,t):
    p=min(1-1e-12,max(1e-12,p));return softmax([math.log(1-p),math.log(p)],t)[1]


def fit_temperature(ps,targets):
    grid=[10**(i/40) for i in range(-40,81)]
    return min(grid,key=lambda t:metrics([scale(p,t) for p in ps],targets)['expected_log_loss'])



def checked_probabilities(q,r):
    keys=list(q['semantics']) if q['question']['type']=='noul' else list(q['question']['criteria'])
    if len(r['logits'])!=len(keys):raise ValueError('Wrong logit count')
    raw=dict(zip(keys,softmax(r['logits'])))
    shipped=dict(zip(keys,softmax(r['logits'],r['shipped_temperature'])))
    answer=r['response']['answers']['answer']
    if q['question']['type']=='noul':
        if not math.isfinite(answer['noul']) or abs(shipped['true']-answer['noul'])>0.000051:raise ValueError('SDK probability mismatch')
    else:
        returned=answer['probabilities']
        if set(returned)!=set(keys) or any(not math.isfinite(returned[k]) or abs(shipped[k]-returned[k])>0.000051 for k in keys):raise ValueError('SDK probability mismatch')
        if answer['choice']!=max(shipped,key=shipped.get):raise ValueError('SDK argmax mismatch')
    return raw,shipped


def report_saved(directory):
    manifest=json.loads((directory/'manifest.json').read_text());requests=manifest['requests']
    if digest(requests)!=manifest['requests_sha256']:raise ValueError('Request hash mismatch')
    records=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    if len(records)!=len(requests) or {r['id'] for r in records}!={r['id'] for r in requests}:raise ValueError('Incomplete or duplicate records')
    by={r['id']:r for r in records};pairs=[]
    for q in requests:
        r=by[q['id']]
        if r['request']!=q or r['status']!='ok':raise ValueError('Failed or mismatched request')
        raw,shipped=checked_probabilities(q,r)
        pairs.append((q,r,raw,shipped))
    result=dict(total=len(records),status='complete',weather={},changed={},rates={})
    for supplied in (False,True):
        subset=[p for p in pairs if p[0]['suite']=='weather' and p[0]['supplied']==supplied]
        cases=defaultdict(list);correct=strict=0
        for q,r,raw,shipped in subset:
            answer=max(shipped,key=shipped.get);cases[q['case']['id']].append(answer)
            correct+=answer==q['target']
            strict+=not q['case']['ambiguous'] and answer==q['case']['expected']
        result['weather'][str(supplied)]=dict(profile_correct=correct,total=len(subset),strict_correct=strict,strict_total=15,stable_cases=sum(len(set(v))==1 for v in cases.values()),answers=dict(cases))
    subset=[p for p in pairs if p[0]['suite']=='changed']
    result['changed']=dict(correct=sum(max(p[3],key=p[3].get)==p[0]['target'] for p in subset),total=len(subset),
        mean_target_probability=sum(p[3][p[0]['target']] for p in subset)/len(subset) if subset else None)
    for family in ('choice','noul'):
        subset=[p for p in pairs if p[0]['suite']=='rate' and p[0]['question']['type']==family]
        if not subset:continue
        cal=[p for p in subset if p[0]['split']=='calibration'];test=[p for p in subset if p[0]['split']=='test']
        cp=[positive_probability(q,raw) for q,r,raw,shipped in cal];cy=[q['target'] for q,r,raw,shipped in cal]
        t=fit_temperature(cp,cy);ys=[q['target'] for q,r,raw,shipped in test]
        rawps=[positive_probability(q,raw) for q,r,raw,shipped in test]
        shippedps=[positive_probability(q,shipped) for q,r,raw,shipped in test]
        fitted=[scale(p,t) for p in rawps];spread=defaultdict(list)
        for p,entry in zip(rawps,test):spread[entry[0]['case_id']].append(p)
        result['rates'][family]=dict(calibration_requests=len(cal),test_requests=len(test),fitted_temperature=t,
            calibration_ids=[p[0]['id'] for p in cal],test_ids=[p[0]['id'] for p in test],
            raw=metrics(rawps,ys),shipped=metrics(shippedps,ys),fitted=metrics(fitted,ys),constant=metrics([.5]*len(ys),ys),
            posterior={name:metrics([posterior(p) for p in ps],[posterior(y) for y in ys]) for name,ps in [('raw',rawps),('shipped',shippedps),('fitted',fitted),('constant',[.5]*len(ys))]},
            max_presentation_range=max(max(ps)-min(ps) for ps in spread.values()),
            predictions=[dict(id=q['id'],target=y,raw=p,shipped=s,fitted=f) for (q,_,_,_),y,p,s,f in zip(test,ys,rawps,shippedps,fitted)])
    corpus=[p for p in pairs if p[0]['suite']=='corpus']
    if corpus:
        result['corpus']={}
        for cohort in ('low','balanced','high'):
            for order in (0,1):
                group=[p for p in corpus if p[0]['cohort']==cohort and p[0]['order']==order]
                answers=[max(p[3],key=p[3].get) for p in group]
                positives=answers.count('billing');truth=sum(p[0]['target']=='billing' for p in group)
                result['corpus'][f'{cohort}/{order}']=dict(total=len(group),correct=sum(a==p[0]['target'] for a,p in zip(answers,group)),
                    billing_count=positives,reference_billing_count=truth,empirical_prior=positives/len(group),
                    beta_posterior_mean=(positives+1)/(len(group)+2),reference_beta_mean=(truth+1)/(len(group)+2))
        by_case=defaultdict(list)
        for q,r,raw,shipped in corpus:by_case[q['case_id']].append(max(shipped,key=shipped.get))
        result['corpus_order_disagreements']=sum(len(set(v))>1 for v in by_case.values())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--report-only',action='store_true');p.add_argument('--deadline');p.add_argument('--followup',action='store_true');args=p.parse_args()
    if args.report_only:write(args.output/'report.json',report_saved(args.output));return
    if not args.deadline:p.error('--deadline required')
    requests=make_followup_requests() if args.followup else make_requests();args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(model=MODEL,revision=REVISION,requests=requests,requests_sha256=digest(requests),
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree=subprocess.check_output(['git','status','--short'],text=True).strip(),
        protocol=(ROOT/'docs/plans/2026-09-28-laya-oracle.md').read_text(),started_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/'manifest.json',manifest)
    from importlib.metadata import version
    import numpy as np
    import torch
    from laya.agent import Agent
    from laya.common import build_sequence,render_options
    if version('laya')!='0.3.21':raise ValueError('Unexpected Laya version')
    torch.manual_seed(0);np.random.seed(0)
    class RecordingAgent(Agent):
        def _forward(self,b):
            logits,act=super()._forward(b);self.observed_logits=logits.tolist();return logits,act
    agent=RecordingAgent(MODEL,revision=REVISION,device='cuda',fast=False,compile=False)
    manifest.update(versions={name:version(name) for name in ('laya','torch','transformers','huggingface-hub','numpy','safetensors')},config=agent.cfg,
        temperature=agent.temperature,temperature_by_options=agent.temperature_by_options,device=str(agent.device),dtype=str(agent.dtype),amp_enabled=agent.amp_enabled)
    write(args.output/'manifest.json',manifest)
    deadline=datetime.fromisoformat(args.deadline).timestamp();schedule=list(requests);random.Random(20260928).shuffle(schedule)
    with (args.output/'records.jsonl').open('x') as stream:
        for i,q in enumerate(schedule):
            if time.time()>=deadline:raise TimeoutError('Inference deadline')
            r=dict(id=q['id'],request=q)
            try:
                internal=agent._to_internal(q['question']);seq,markers,stats=build_sequence(agent.tok,q['state'],internal,512,192,return_stats=True)
                full,_,_=build_sequence(agent.tok,q['state'],internal,4096,2048,return_stats=True)
                if seq!=full or any(len(agent.tok.encode(' '+o,add_special_tokens=False))>48 for o in render_options(internal)):
                    raise ValueError('Input would be truncated')
                start=time.perf_counter();response=agent.predict(q['state'],{'answer':q['question']});elapsed=time.perf_counter()-start
                logits=agent.observed_logits[0][:len(markers)]
                from laya.common import QTYPES,temp_bucket
                qt=QTYPES[internal['t']];temperature=agent.temperature_by_options.get(temp_bucket(qt,len(markers)),agent.temperature[qt])
                r.update(status='ok',response=response,logits=logits,shipped_temperature=temperature,token_ids=seq,markers=markers,option_stats=stats,seconds=elapsed,device=str(agent.device))
            except Exception as exc:r.update(status='error',error=f'{type(exc).__name__}: {exc}')
            stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            if (i+1)%10==0:print(f'Completed {i+1}/{len(requests)}',flush=True)
    write(args.output/'report.json',report_saved(args.output))


if __name__=='__main__':main()
