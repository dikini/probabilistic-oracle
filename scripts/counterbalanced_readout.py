"""Crossed category-letter and display-order diagnostic, without new fitting."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
from statistics import mean,pstdev
import subprocess
import time
import categorical_readout as c
from probabilistic_oracle.oracle import ScoreRequest

PROTOCOL=Path('docs/plans/2026-09-24-counterbalanced-readout.md')


def layout(n,mapping,order):
    if n not in (3,5) or not 0<=mapping<n or not 0<=order<n:raise ValueError('Invalid layout')
    return [chr(65+(i+mapping)%n) for i in range(n)],[(j+order)%n for j in range(n)]


def semantic_logs(letter_logs,n,mapping):
    if len(letter_logs)!=n:raise ValueError('Wrong candidate count')
    return [letter_logs[(i+mapping)%n] for i in range(n)]


def prompt_text(request,n,mapping,order):
    labels,display=layout(n,mapping,order)
    old='\n'.join(f'{chr(65+i)} = {category}' for i,category in enumerate(c.CATEGORIES[n]))
    new='\n'.join(f'{labels[i]} = {c.CATEGORIES[n][i]}' for i in display)
    return c.prompt_text(request,n).replace(old,new,1)


def prepare(tokenizer,request,n,mapping,order):
    prompt=tokenizer.apply_chat_template([{'role':'user','content':prompt_text(request,n,mapping,order)}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer.encode(prompt,add_special_tokens=False);candidates=[]
    for i in range(n):
        label=chr(65+i);token=tokenizer.encode(label,add_special_tokens=False)
        if len(token)!=1 or tokenizer.encode(prompt+label,add_special_tokens=False)!=ids+token:
            raise ValueError('Invalid token boundary')
        candidates+=token
    if len(set(candidates))!=n:raise ValueError('Duplicate tokens')
    return prompt,ids,candidates


def fresh_cases(existing):
    rng=random.Random(2026092404);seen={tuple(x) for x in existing};cases=[]
    while len(cases)<4:
        cuts=[0]+sorted(rng.sample(range(1,100),3))+[100];counts=[b-a for a,b in zip(cuts,cuts[1:])]
        if tuple(counts) in seen:continue
        seen.add(tuple(counts));cases.append(dict(id=f'counter-{len(cases)+1:02}',split='eval',counts=counts))
    return cases


def make_queries(cases):
    queries=[asdict(q) for case in cases for q in c.synthetic_queries(case,('A','B')) if q.variant=='base' and q.kind in ('h','not_h','joint')]
    for color,target in [('red',1),('blue',0)]:
        request=ScoreRequest('All 100 cards in this deck are red. One card is selected uniformly at random.',f'The selected card is {color}.')
        queries.append(dict(id=f'control-{color}',case_id=f'control-{color}',split='control',variant='base',kind='h',target=target,request=asdict(request)))
    return json.loads(json.dumps(queries))


def grid(queries):
    rows=[dict(id=q['id'],scale=n,mapping=a,order=b) for q in queries for n in (3,5) for a in range(n) for b in range(n)]
    random.Random(2026092404).shuffle(rows)
    return rows


def key(r):return r['id'],r['scale'],r['mapping'],r['order']


def validate(manifest,records):
    expected=[key(r) for r in grid(manifest['queries'])];actual=[key(r) for r in records]
    if len(expected)!=len(set(expected)) or len(actual)!=len(set(actual)) or set(expected)!=set(actual):
        raise ValueError('Incomplete, duplicate or unexpected observation grid')
    queries={q['id']:q for q in manifest['queries']}
    for r in records:
        if r['status']!='ok':raise ValueError(f"Failed observation: {key(r)}")
        n=r['scale'];labels,display=layout(n,r['mapping'],r['order'])
        if r['category_letters']!=labels or r['display_order']!=display:raise ValueError('Layout mismatch')
        if r['request']!=queries[r['id']]['request'] or r['provenance']!=manifest['backend']:raise ValueError('Provenance/request mismatch')
        if len(r['candidate_token_ids'])!=n or len(set(r['candidate_token_ids']))!=n:raise ValueError('Invalid token IDs')
        if r['prompt_text']!=prompt_text(ScoreRequest(**r['request']),n,r['mapping'],r['order']):raise ValueError('Prompt mismatch')
        logs=semantic_logs(r['letter_logprobs'],n,r['mapping']);p,mass,_=c.score_logs(logs,c.anchors(n))
        if not math.isclose(p,r['raw_score'],abs_tol=1e-12) or not math.isclose(mass,r['candidate_mass'],abs_tol=1e-12):raise ValueError('Score or mass mismatch')
    return records


def diagnostics(rows,n):
    category=Counter();letter=Counter();position=Counter();cm=[0]*n;lm=[0]*n;pm=[0]*n
    conditional={k:dict(chosen=0,total=0) for k in ('a_when_not_first','first_when_not_a','true_when_neither_a_nor_first')}
    scores=[];ties=0
    for r in rows:
        labels,display=layout(n,r['mapping'],r['order']);logs=semantic_logs(r['letter_logprobs'],n,r['mapping'])
        p,_,weights=c.score_logs(logs,c.anchors(n));scores.append(p)
        chosen=max(range(n),key=lambda i:logs[i]);ties+=sum(x==max(logs) for x in logs)>1
        category[c.CATEGORIES[n][chosen]]+=1;letter[labels[chosen]]+=1;position[str(display.index(chosen)+1)]+=1
        for i,w in enumerate(weights):cm[i]+=w;lm[ord(labels[i])-65]+=w;pm[display.index(i)]+=w
        first=display[0]
        if labels[first]!='A':
            conditional['a_when_not_first']['total']+=1;conditional['a_when_not_first']['chosen']+=labels[chosen]=='A'
            conditional['first_when_not_a']['total']+=1;conditional['first_when_not_a']['chosen']+=chosen==first
        if labels[0]!='A' and first!=0:
            conditional['true_when_neither_a_nor_first']['total']+=1;conditional['true_when_neither_a_nor_first']['chosen']+=chosen==0
    count=len(rows)
    return dict(count=count,category_counts=dict(category),letter_counts=dict(letter),position_counts=dict(position),
        mean_category_weights=dict(zip(c.CATEGORIES[n],[x/count for x in cm])),
        mean_letter_weights=dict(zip([chr(65+i) for i in range(n)],[x/count for x in lm])),
        mean_position_weights=dict(zip([str(i+1) for i in range(n)],[x/count for x in pm])),
        ties=ties,mean_raw_score=mean(scores),**conditional)


def summarize(manifest,records):
    if c.digest(manifest['queries'])!=manifest['query_sha256'] or c.digest(manifest['protocol'])!=manifest['protocol_sha256'] or c.digest(manifest['frozen_calibration'])!=manifest['calibration_sha256']:
        raise ValueError('Manifest hash mismatch')
    validate(manifest,records);queries=manifest['queries'];report=dict(total_requests=len(records),methods={},scope='formatting_diagnostic_no_full_gate_verdict')
    for n in (3,5):
        result=dict(per_query={},frozen_calibration=manifest['frozen_calibration'][str(n)])
        fit=result['frozen_calibration']
        for q in queries:
            rows=[r for r in records if r['id']==q['id'] and r['scale']==n]
            raw=[c.score_logs(semantic_logs(r['letter_logprobs'],n,r['mapping']),c.anchors(n))[0] for r in rows]
            calibrated=[c.score_logs(semantic_logs(r['letter_logprobs'],n,r['mapping']),c.anchors(n),fit['temperature'],fit['bias'])[0] for r in rows]
            canonical=next(i for i,r in enumerate(rows) if r['mapping']==r['order']==0)
            result['per_query'][q['id']]=dict(target=q['target'],raw_canonical=raw[canonical],raw_balanced=mean(raw),
                calibrated_canonical=calibrated[canonical],calibrated_balanced=mean(calibrated),
                raw_sd=pstdev(raw),raw_range=max(raw)-min(raw),diagnostics=diagnostics(rows,n))
        synthetic_ids={q['id'] for q in queries if q['split']=='eval'}
        synthetic=[r for r in records if r['id'] in synthetic_ids and r['scale']==n]
        result['synthetic_diagnostics']=diagnostics(synthetic,n)
        result['candidate_mass']=dict(mean=mean(r['candidate_mass'] for r in synthetic),minimum=min(r['candidate_mass'] for r in synthetic))
        result['generated_in_candidates']=dict(chosen=sum(bool(r['generated_token_ids']) and r['generated_token_ids'][0] in r['candidate_token_ids'] for r in synthetic),total=len(synthetic))
        result['mean_query_raw_sd']=mean(result['per_query'][q]['raw_sd'] for q in synthetic_ids)
        result['mean_query_raw_range']=mean(result['per_query'][q]['raw_range'] for q in synthetic_ids)
        result['accuracy']={}
        for mode in ('raw_canonical','raw_balanced','calibrated_canonical','calibrated_balanced','constant_half','previous_calibration_mean'):
            def score(q):
                if mode=='constant_half':return .5
                if mode=='previous_calibration_mean':return manifest['fitted_constant']
                return result['per_query'][q['id']][mode]
            errors=[c.quality([(score(q),q['target']) for q in queries if q['case_id']==case])['rmse']
                    for case in sorted({q['case_id'] for q in queries if q['split']=='eval'})]
            result['accuracy'][mode]=dict(mean_world_rmse=mean(errors),world_rmse=errors)
        report['methods'][str(n)]=result
    return report


def report_saved(directory):
    manifest=json.loads((directory/'manifest.json').read_text());records=[json.loads(x) for x in (directory/'records.jsonl').read_text().splitlines()]
    return summarize(manifest,records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--history',type=Path)
    p.add_argument('--deadline');p.add_argument('--report-only',action='store_true');args=p.parse_args()
    if args.report_only:
        report=report_saved(args.output);c.write(args.output/'report.json',report);print(json.dumps(report));return
    if not args.history or not args.deadline:p.error('--history and --deadline required')
    deadline=datetime.fromisoformat(args.deadline).timestamp()
    old=json.loads(Path('bench/cases/synthetic.json').read_text())+json.loads((args.history/'parameter-sweep-20260924/fresh-cases.json').read_text())
    for name in ('readout-comparison-20260924','categorical-readout-20260924'):
        old+=json.loads((args.history/name/'manifest.json').read_text())['cases']
    cases=fresh_cases([x['counts'] for x in old]);queries=make_queries(cases);protocol=PROTOCOL.read_text()
    frozen=json.loads((args.history/'categorical-readout-20260924/calibration.json').read_text())
    constant=json.loads((args.history/'categorical-readout-20260924/report.json').read_text())['baselines']['calibration_mean']['score']
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(cases=cases,queries=queries,query_sha256=c.digest(queries),protocol=protocol,protocol_sha256=c.digest(protocol),
        frozen_calibration=frozen,calibration_sha256=c.digest(frozen),fitted_constant=constant,excluded_counts=[x['counts'] for x in old],
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        working_tree=subprocess.check_output(['git','status','--short'],text=True).strip(),started_utc=datetime.now(timezone.utc).isoformat())
    c.write(args.output/'manifest.json',manifest)
    from probabilistic_oracle.backends.vllm import VllmOracle
    oracle=VllmOracle();manifest['backend']=dict(oracle.provenance,prompt_template='counterbalanced-categorical-v1')
    c.write(args.output/'manifest.json',manifest);by={q['id']:q for q in queries}
    with (args.output/'records.jsonl').open('x') as f:
        for i,cell in enumerate(grid(queries)):
            if time.time()>=deadline:raise TimeoutError('Deadline reached; incomplete records retained')
            r=dict(cell);n=r['scale'];mapping=r['mapping'];order=r['order'];q=by[r['id']];request=ScoreRequest(**q['request'])
            try:
                prompt,ids,candidates=prepare(oracle.tokenizer,request,n,mapping,order)
                if len(ids)+1>oracle.engine['max_model_len']:raise ValueError('Prompt too long')
                start=time.perf_counter();out=oracle.llm.generate([{'prompt_token_ids':ids}],oracle.params,use_tqdm=False)[0].outputs[0]
                if not out.logprobs or any(t not in out.logprobs[0] for t in candidates):raise ValueError('Missing candidate logs')
                logs=[out.logprobs[0][t].logprob for t in candidates];score,mass,_=c.score_logs(semantic_logs(logs,n,mapping),c.anchors(n))
                labels,display=layout(n,mapping,order)
                r.update(status='ok',request=q['request'],prompt=prompt,prompt_text=prompt_text(request,n,mapping,order),prompt_token_ids=ids,
                    candidate_token_ids=candidates,letter_logprobs=logs,category_letters=labels,display_order=display,
                    raw_score=score,candidate_mass=mass,generated_token_ids=list(out.token_ids),text=out.text,seconds=time.perf_counter()-start,
                    finish_reason=out.finish_reason,provenance=manifest['backend'])
            except Exception as exc:r.update(status='error',error=f'{type(exc).__name__}: {exc}')
            f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
            if (i+1)%34==0:print(f'Completed {i+1}/476 requests',flush=True)
    report=report_saved(args.output);c.write(args.output/'report.json',report);print('Complete report saved',flush=True)


if __name__=='__main__':main()
