"""Commands for validating cases, running the GPU experiment, and reporting."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .bench import load_queries, run_benchmark
from .metrics import summarize

VERBALIZERS = {'true-false':('TRUE','FALSE'), 'yes-no':('YES','NO'), 'a-b':('A','B')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('validate','run'):
        p=commands.add_parser(name)
        p.add_argument('--synthetic',type=Path,default=Path('bench/cases/synthetic.json'))
        p.add_argument('--world',type=Path,default=Path('bench/cases/world.json'))
        p.add_argument('--verbalizers',choices=VERBALIZERS,default='true-false')
        if name=='run':
            p.add_argument('--output',type=Path,required=True)
            p.add_argument('--protocol',type=Path,default=Path('docs/experiment-protocol.md'))
    p=commands.add_parser('report')
    p.add_argument('directory',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='report':
            report=summarize(args.directory)
            (args.directory/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps(report,indent=2,allow_nan=False))
            return 2 if report['decision']=='incomplete' else 0
        queries=load_queries(args.synthetic,args.world,VERBALIZERS[args.verbalizers])
        if args.command=='validate':
            counts={split:len({q.case_id for q in queries if q.split==split}) for split in ('calibration','eval','world')}
            print(json.dumps(dict(queries=len(queries),cases_by_split=counts)))
            return 0
        if args.output.exists():
            raise ValueError(f'Run directory already exists: {args.output}')
        protocol=args.protocol.read_text()
        metadata=dict(protocol=protocol,protocol_sha256=hashlib.sha256(protocol.encode()).hexdigest(),
                      verbalizer_set=args.verbalizers)
        from .backends.vllm import VllmOracle
        oracle=VllmOracle()
        run_benchmark(queries,oracle,args.output,metadata)
        report=summarize(args.output)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(args.output),decision=report['decision'],errors=len(report['errors']))))
        return 2 if report['decision']=='incomplete' else 0
    except (ValueError, OSError, KeyError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
