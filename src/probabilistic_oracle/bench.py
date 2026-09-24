"""Finite-world queries and append-only experiment records."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

from .oracle import ScoreRequest


@dataclass(frozen=True)
class Query:
    id: str
    case_id: str
    split: str
    variant: str
    kind: str
    request: ScoreRequest
    target: float | None


VARIANTS = ('base', 'paraphrase', 'irrelevant')
H = 'The selected card is red.'
NOT_H = 'The selected card is not red.'
E = 'The selected card has a circle.'


def synthetic_queries(case: dict, verbalizers=('TRUE', 'FALSE')) -> list[Query]:
    counts = case['counts']
    if len(counts) != 4 or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('Counts must be four nonnegative integers')
    a, b, c, d = counts
    n = sum(counts)
    if not n or not a + b or not c + d or not a + c:
        raise ValueError('World must support both hypotheses and the observation')
    if case['split'] not in ('calibration', 'eval'):
        raise ValueError('Unknown synthetic split')
    targets = dict(h=Fraction(a+b, n), not_h=Fraction(c+d, n), e_h=Fraction(a, a+b),
                   e_not_h=Fraction(c, c+d), posterior=Fraction(a, a+c), joint=Fraction(a, n))
    original = (f'A deck contains {n} cards: {a} red cards with circles, {b} red cards without circles, '
                f'{c} blue cards with circles, and {d} blue cards without circles. '
                'One card is selected uniformly at random. These categories exhaust the deck.')
    paraphrase = (f'Choose one card at random, with equal chance for each of {n} cards. '
                  f'The complete inventory is: red/circle {a}; red/no circle {b}; '
                  f'blue/circle {c}; blue/no circle {d}.')
    result = []
    for variant in VARIANTS:
        context = paraphrase if variant == 'paraphrase' else original
        if variant == 'irrelevant':
            context += ' The inventory is printed in a monospaced font; printing does not affect selection.'
        requests = dict(
            h=ScoreRequest(context, H, verbalizers=verbalizers),
            not_h=ScoreRequest(context, NOT_H, verbalizers=verbalizers),
            e_h=ScoreRequest(context, E, assumptions=(H,), verbalizers=verbalizers),
            e_not_h=ScoreRequest(context, E, assumptions=(NOT_H,), verbalizers=verbalizers),
            posterior=ScoreRequest(context, H, observations=(E,), verbalizers=verbalizers),
            joint=ScoreRequest(context, 'The selected card is red and has a circle.', verbalizers=verbalizers),
        )
        for kind, request in requests.items():
            result.append(Query(f"{case['id']}/{variant}/{kind}", case['id'], case['split'],
                                variant, kind, request, float(targets[kind])))
    return result


def load_queries(synthetic: Path, world: Path | None = None, verbalizers=('TRUE', 'FALSE')) -> list[Query]:
    result, seen = [], set()
    sources = [(json.loads(Path(synthetic).read_text()), False)]
    if world is not None:
        sources.append((json.loads(Path(world).read_text()), True))
    for cases, is_world in sources:
        for case in cases:
            case_id = case['id']
            if not isinstance(case_id, str) or not case_id or '/' in case_id:
                raise ValueError('Case ID must be a nonempty string without slashes')
            if case_id in seen:
                raise ValueError(f'Duplicate case ID: {case_id}')
            seen.add(case_id)
            if not is_world:
                result.extend(synthetic_queries(case, verbalizers))
                continue
            for variant in VARIANTS:
                context = case['paraphrase_context'] if variant == 'paraphrase' else case['context']
                if variant == 'irrelevant':
                    context += ' This question is displayed in a monospaced font.'
                for kind, proposition in [('h', case['proposition']), ('not_h', case['negation'])]:
                    result.append(Query(f'{case_id}/{variant}/{kind}', case_id, 'world', variant, kind,
                                        ScoreRequest(context, proposition, verbalizers=verbalizers), None))
    if not result:
        raise ValueError('No benchmark cases')
    return result


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_benchmark(queries: list[Query], oracle, directory: Path, metadata: dict | None = None):
    if not queries or len({q.id for q in queries}) != len(queries):
        raise ValueError('Queries must have unique IDs and must not be empty')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = [asdict(q) for q in queries]
    manifest = dict(schema_version=1, started_utc=datetime.now(timezone.utc).isoformat(),
                    backend=oracle.provenance, code_revision=_git('rev-parse', 'HEAD'),
                    working_tree=_git('status', '--short'), queries=encoded,
                    query_sha256=hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest(),
                    metadata=metadata or {})
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    with (directory/'records.jsonl').open('x') as f:
        for query in queries:
            try:
                record = dict(id=query.id, status='ok', result=asdict(oracle.score(query.request)))
                line = json.dumps(record, allow_nan=False)
            except Exception as exc:
                record = dict(id=query.id, status='error', error=f'{type(exc).__name__}: {exc}')
                line = json.dumps(record)
            f.write(line+'\n')
            f.flush()


def read_run(directory: Path) -> tuple[dict, list[dict]]:
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported run schema')
    digest = hashlib.sha256(json.dumps(manifest['queries'], sort_keys=True).encode()).hexdigest()
    if digest != manifest.get('query_sha256'):
        raise ValueError('Query manifest hash mismatch')
    records = [json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    expected = [q['id'] for q in manifest['queries']]
    actual = [r['id'] for r in records]
    if len(expected) != len(set(expected)) or len(actual) != len(set(actual)) or set(expected) != set(actual):
        raise ValueError('Incomplete run or duplicate/unexpected query IDs')
    if any(r.get('status') not in ('ok', 'error') for r in records):
        raise ValueError('Invalid record status')
    return manifest, records
