"""Recheck a saved sweep and inspect the calibration grid on development data only."""

import argparse
import json
from pathlib import Path
from statistics import mean

import parameter_sweep as sweep
from probabilistic_oracle.metrics import _case_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=args.directory
    original=json.loads((root/'leaderboard.json').read_text())
    reviewed=[]
    diagnostic=[]
    for candidate in original:
        name=candidate['config']['name']
        path=Path(candidate.get('source_run',str(root/name)))
        rows=sweep.load_rows(path)
        rows=[(q,p) for q,p in rows if (q['split']=='calibration' and q['variant']=='base')
              or q['case_id'] in {'eval-01','eval-02','eval-03','eval-04'}]
        result=sweep.analyze(rows,candidate['config'])
        result['source_run']=str(path)
        reviewed.append(result)
        groups={}
        for q,p in rows:
            if q['split']=='eval':
                groups.setdefault(q['case_id'],[]).append((q,p))
        trials=[]
        for t in sweep.TEMPERATURES:
            for b in sweep.BIASES:
                cases=[_case_metrics(v,lambda p:sweep.transform(p,t,b)) for v in groups.values()]
                metrics={k:mean(c[k] for c in cases) for k in sweep.LIMITS}
                trials.append(dict(temperature=t,bias=b,metrics=metrics,
                                   passes_numeric_gates=all(metrics[k]<=v for k,v in sweep.LIMITS.items())))
        diagnostic.append(dict(config=name,role='optimistic development-only diagnostic; not selection',trials=trials))
    sweep.write(root/'reviewed-leaderboard.json',reviewed)
    sweep.write(root/'development-grid-diagnostic.json',diagnostic)
    if (root/'final-report.json').exists():
        original_report=json.loads((root/'final-report.json').read_text())
        winner=min(reviewed,key=sweep.rank)
        if winner['config']!=original_report['selection']['config'] or winner['calibration']!=original_report['selection']['calibration']:
            raise ValueError('Reviewed development selection differs: fresh evaluation cannot validate the corrected selection')
        rows=sweep.load_rows(root/'fresh-evaluation')
        calibrated=sweep.measurements(rows,winner['calibration'])
        raw=sweep.measurements(rows,fitted_constant=winner['calibration']['fitted_constant'])
        checks=sweep.gates(calibrated,raw)
        report=dict(original_report,raw=raw,calibrated=calibrated,gates=checks,
                    decision='thresholds_met' if all(checks.values()) else 'thresholds_not_met',
                    review_note='Rechecked both raw and calibrated undefined paths; selection unchanged.')
        sweep.write(root/'reviewed-final-report.json',report)
        print('Fresh evaluation:',report['decision'])
    for item in diagnostic:
        print(item['config'],'grid trials',len(item['trials']),
              'minimum development RMSE',round(min(t['metrics']['rmse'] for t in item['trials']),6),
              'numeric gate passes',sum(t['passes_numeric_gates'] for t in item['trials']))


if __name__=='__main__':
    main()
