"""Summarize request metadata from a server log; never print request bodies.

An entire log may span several submissions. --last 600 selects the last 600
matching completed-request log lines; it does not establish submission bounds.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

LINE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} (tutor|guard) '
                  r'(\d+(?:\.\d+)?)s turns=\d+ chars=\d+ errors=([^\s]+)$')
ERROR = re.compile(r'[A-Za-z_][A-Za-z_0-9]*:[A-Za-z_][A-Za-z_0-9]*\Z')


def summarize(text, last=None):
    rows=[]
    for line in text.splitlines():
        match=LINE.fullmatch(line.strip())
        if match:
            endpoint,seconds,errors=match.groups()
            rows.append((endpoint,float(seconds),errors))
    total=len(rows)
    if last is not None:
        if last<=0:raise ValueError('last must be positive')
        rows=rows[-last:]
    result={'scope':'completed_request_metadata_not_official_score_or_submission_boundaries',
            'matched_in_file':total,'selected_requests':len(rows),'endpoints':{}}
    for endpoint in ('tutor','guard'):
        selected=[r for r in rows if r[0]==endpoint]
        if not selected:continue
        durations=sorted(r[1] for r in selected)
        errors=Counter(e for _,_,raw in selected for e in raw.split(',') if ERROR.fullmatch(e))
        result['endpoints'][endpoint]={'requests':len(selected),
            'internal_failure_requests':sum(raw!='-' for _,_,raw in selected),
            'timeout_requests':sum(any(word in raw for word in ('Timeout','Deadline','NoBudget')) for _,_,raw in selected),
            'p50_seconds':statistics.median(durations),
            'p95_seconds':durations[min(len(durations)-1,int(.95*len(durations)))],
            'max_seconds':max(durations),'stage_errors':dict(sorted(errors.items()))}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log',type=Path)
    ap.add_argument('--last',type=int)
    args=ap.parse_args()
    if args.last is not None and args.last<=0:ap.error('--last must be positive')
    print(json.dumps(summarize(args.log.read_text(errors='replace'),args.last),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
