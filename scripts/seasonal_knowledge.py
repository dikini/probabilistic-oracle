"""Broad seasonal knowledge pilot with vLLM constrained categorical output."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

CHOICES=['cold','mild','hot','unknown']
VARIANTS=['neutral','question_first','emphasis']
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/plans/2026-09-28-seasonal-knowledge.md'


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_cases():return json.loads((ROOT/'bench/cases/seasonal-knowledge.json').read_text())


def queries(cases,facts=None):
    if facts is not None and any(c['country'] not in facts or not facts[c['country']].get('text') for c in cases):
        raise ValueError('Missing country facts')
    result=[]
    for case in cases:
        for variant in VARIANTS:
            country=case['country'];month=case['month']
            scope='Consider the country as a whole, allowing for regional variation.'
            question=f'Which category best describes typical daytime temperatures in {country} in {month}?'
            if variant=='question_first':text=question+'\n'+scope
            else:
                if variant=='emphasis':question=f'Which category best describes typical daytime temperatures in {country.upper()} in {month.upper()}?'
                text=scope+'\n'+question
            text+='\nAnswer with exactly one word: cold, mild, hot, or unknown. Use unknown if you cannot select a category from your general knowledge. Do not explain.'
            if facts is not None:
                text='Background facts for this question (take as given):\n'+facts[country]['text']+'\n\n'+text
            q=dict(id=case['id']+'/'+variant,case=case,variant=variant,text=text)
            if facts is not None and 'provided_labels' in facts[country]:
                q['provided_label']=facts[country]['provided_labels'][month]
            result.append(q)
    return result


def parse(text,finish_reason):
    answer=text.strip()
    if finish_reason!='stop' or answer not in CHOICES:raise ValueError('Invalid or incomplete categorical answer')
    return answer


def summarize(qs,records):
    expected=[q['id'] for q in qs];actual=[r['id'] for r in records]
    if len(set(expected))!=len(expected) or len(set(actual))!=len(actual) or set(expected)!=set(actual):
        raise ValueError('Incomplete, duplicate or unexpected records')
    by={r['id']:r for r in records};answers={};errors=[]
    for q in qs:
        r=by[q['id']]
        try:
            if r['status']!='ok':raise ValueError(r.get('error','failed request'))
            answer=parse(r['text'],r['finish_reason'])
            if answer!=r['answer']:raise ValueError('Stored answer differs from raw text')
            answers[q['id']]=answer
        except (ValueError,KeyError,TypeError) as exc:errors.append(dict(id=q['id'],error=str(exc)))
    report=dict(status='complete' if not errors else 'incomplete',total=len(qs),valid=len(answers),errors=errors)
    if errors:return report
    groups={}
    for q in qs:groups.setdefault(q['case']['id'],[]).append(q)
    strict=[q for q in qs if not q['case']['ambiguous']]
    report.update(answer_counts=dict(Counter(answers.values())),unknown_count=sum(a=='unknown' for a in answers.values()),
        strict=dict(correct=sum(answers[q['id']]==q['case']['expected'] for q in strict),total=len(strict)),
        constant_cold_baseline=dict(correct=sum(q['case']['expected']=='cold' for q in strict),total=len(strict)),
        strict_cases_correct_all=sum(all(answers[q['id']]==q['case']['expected'] for q in group) for group in groups.values() if not group[0]['case']['ambiguous']),
        stable_cases=sum(len({answers[q['id']] for q in group})==1 for group in groups.values()),
        pairwise_disagreements=sum(answers[a['id']]!=answers[b['id']] for group in groups.values() for a,b in itertools.combinations(group,2)),
        cases={key:dict(country=group[0]['case']['country'],month=group[0]['case']['month'],expected=group[0]['case']['expected'],ambiguous=group[0]['case']['ambiguous'],
                       answers={q['variant']:answers[q['id']] for q in group}) for key,group in groups.items()})
    report['seasonal']=[];report['singapore_hot_pairs']=0
    for country in dict.fromkeys(q['case']['country'] for q in qs):
        for variant in VARIANTS:
            jan=answers[f'{country.lower()}-january/{variant}'];jul=answers[f'{country.lower()}-july/{variant}']
            if country=='Singapore':
                report['singapore_hot_pairs']+=jan==jul=='hot';continue
            if 'unknown' in (jan,jul):outcome='unknown'
            elif jan==jul:outcome='tied'
            else:
                warmer=next(q['case']['warmer_month'] for q in qs if q['case']['country']==country)
                july_warmer=CHOICES.index(jul)>CHOICES.index(jan)
                outcome='expected' if july_warmer==(warmer=='July') else 'reversed'
            report['seasonal'].append(dict(country=country,variant=variant,january=jan,july=jul,outcome=outcome))
    report['seasonal_counts']=dict(Counter(x['outcome'] for x in report['seasonal']))
    supplied=[q for q in qs if 'provided_label' in q]
    if supplied:
        report['supplied_knowledge_agreement']=dict(correct=sum(answers[q['id']]==q['provided_label'] for q in supplied),total=len(supplied))
    return report


def compare_reports(baseline,current):
    if baseline['status']!='complete' or current['status']!='complete':
        return dict(status='incomplete')
    if set(baseline['cases'])!=set(current['cases']):raise ValueError('Unmatched baseline cases')
    changed=0
    for case,row in current['cases'].items():
        old=baseline['cases'][case]
        if any(row[k]!=old[k] for k in ('country','month','expected','ambiguous')):raise ValueError('Changed case references')
        changed+=sum(row['answers'][v]!=old['answers'][v] for v in VARIANTS)
    return dict(changed_answers=changed,strict_correct_delta=current['strict']['correct']-baseline['strict']['correct'],
        unknown_delta=current['unknown_count']-baseline['unknown_count'],stable_cases_delta=current['stable_cases']-baseline['stable_cases'])


def report_saved(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    if digest(manifest['queries'])!=manifest['query_sha256'] or digest(manifest['protocol'])!=manifest['protocol_sha256']:
        raise ValueError('Manifest hash mismatch')
    records=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    by={q['id']:q for q in manifest['queries']}
    for r in records:
        if r['status']=='ok' and (r['request_text']!=by[r['id']]['text'] or r['provenance']!=manifest['backend']):
            raise ValueError('Prompt or provenance mismatch')
    report=summarize(manifest['queries'],records)
    if 'supplied_facts' in manifest:
        if digest(manifest['supplied_facts'])!=manifest['facts_sha256'] or digest(manifest['baseline_report'])!=manifest['baseline_report_sha256']:
            raise ValueError('Facts/baseline hash mismatch')
        report['comparison_with_no_facts']=compare_reports(manifest['baseline_report'],report)
    return report


def validate_baseline_backend(manifest, engine, sampling):
    backend = manifest.get('backend', {})
    if backend.get('engine') != engine or backend.get('sampling') != sampling:
        raise ValueError('Baseline backend must match the selected model, engine and sampling')
    return backend


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline');p.add_argument('--report-only',action='store_true');p.add_argument('--facts',action='store_true');p.add_argument('--baseline',type=Path);p.add_argument('--model-profile',choices=('qwen3-0.6b','qwen3-4b-awq'),default='qwen3-0.6b');args=p.parse_args()
    if args.report_only:
        report=report_saved(args.output);write(args.output/'report.json',report);print(json.dumps(report));return
    if not args.deadline:p.error('--deadline required')
    if args.facts and not args.baseline:p.error('--facts requires --baseline')
    facts=json.loads((ROOT/'bench/cases/seasonal-facts.json').read_text()) if args.facts else None
    deadline=datetime.fromisoformat(args.deadline).timestamp();qs=queries(load_cases(),facts)
    protocol=(ROOT/'docs/plans/2026-09-28-seasonal-facts.md' if args.facts else PROTOCOL).read_text()
    if args.model_profile != 'qwen3-0.6b':
        protocol += '\n\n' + (ROOT/'docs/plans/2026-09-28-larger-seasonal.md').read_text()
    args.output.mkdir(parents=True,exist_ok=False)
    settings=dict(temperature=0,max_tokens=16,seed=0,structured_outputs=dict(choice=CHOICES))
    manifest=dict(queries=qs,query_sha256=digest(qs),protocol=protocol,protocol_sha256=digest(protocol),sampling=settings,
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree=subprocess.check_output(['git','status','--short'],text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat())
    if args.facts:
        baseline=report_saved(args.baseline)
        if baseline['status']!='complete':raise ValueError('Incomplete baseline')
        baseline_manifest=json.loads((args.baseline/'manifest.json').read_text())
        if 'supplied_facts' in baseline_manifest:raise ValueError('Baseline must be no-facts run')
        from probabilistic_oracle.backends.vllm import engine_config
        manifest['baseline_backend']=validate_baseline_backend(baseline_manifest,engine_config(args.model_profile),settings)
        manifest.update(supplied_facts=facts,facts_sha256=digest(facts),baseline_report=baseline,baseline_report_sha256=digest(baseline),
                        baseline_directory=str(args.baseline),baseline_query_sha256=baseline_manifest['query_sha256'])
    write(args.output/'manifest.json',manifest)
    from probabilistic_oracle.backends.vllm import VllmOracle
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    oracle=VllmOracle(profile=args.model_profile)
    params=SamplingParams(temperature=0,max_tokens=16,seed=0,structured_outputs=StructuredOutputsParams(choice=CHOICES))
    backend=dict(oracle.provenance,sampling=settings,prompt_template='seasonal-facts-v1' if args.facts else 'seasonal-knowledge-v1',
                 structured_output_config=asdict(oracle.llm.llm_engine.vllm_config.structured_outputs_config))
    manifest['backend']=backend;write(args.output/'manifest.json',manifest)
    schedule=list(qs);random.Random(2026092801).shuffle(schedule)
    with (args.output/'records.jsonl').open('x') as f:
        for i,q in enumerate(schedule):
            if time.time()>=deadline:raise TimeoutError('Deadline reached; incomplete records retained')
            r=dict(id=q['id'])
            try:
                prompt=oracle.tokenizer.apply_chat_template([dict(role='user',content=q['text'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=oracle.tokenizer.encode(prompt,add_special_tokens=False)
                if len(ids)+16>oracle.engine['max_model_len']:raise ValueError('Prompt exceeds context budget')
                start=time.perf_counter();out=oracle.llm.generate([dict(prompt_token_ids=ids)],params,use_tqdm=False)[0].outputs[0]
                r.update(request_text=q['text'],prompt=prompt,prompt_token_ids=ids,text=out.text,finish_reason=out.finish_reason,
                         generated_token_ids=list(out.token_ids),seconds=time.perf_counter()-start,provenance=backend)
                r.update(answer=parse(out.text,out.finish_reason),status='ok')
            except Exception as exc:r.update(status='error',error=f'{type(exc).__name__}: {exc}')
            f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
            print(f'Completed {i+1}/36',flush=True)
    report=report_saved(args.output);write(args.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
