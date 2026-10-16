"""Endpoint-level overview of recorded criteria; no diagnostic aggregation."""
from collections import defaultdict


def endpoint_reviews(payload):
    groups = defaultdict(list)
    findings = payload.get('research_findings', [])
    for finding in findings if isinstance(findings, list) else []:
        if isinstance(finding, dict) and isinstance(finding.get('endpoint_id'), str) and finding['endpoint_id']:
            groups[finding['endpoint_id']].append(finding)

    def summarize(items):
        states = set(items)
        if not states or states == {'unknown'}:
            return 'unknown'
        if 'positive' in states and 'negative' in states:
            return 'mixed'
        if 'unknown' in states or 'gray_zone' in states:
            return 'partial'
        return next(iter(states))

    labels = {'unknown': 'Nicht beurteilbar', 'mixed': 'Gemischte Kriterien',
              'partial': 'Unvollständig / Grauzone', 'positive': 'Kriterien positiv',
              'negative': 'Kriterien negativ (kein Krankheitsausschluss)'}
    output = []
    for endpoint, rows in sorted(groups.items()):
        research, classical = set(), set()
        details = []
        for row in rows:
            state = row.get('classification') if row.get('applicable') is True else 'unknown'
            state = state if state in ('positive', 'negative', 'gray_zone') else 'unknown'
            research.add(state)
            details.append(f"{row.get('criterion_id', '?')}: {state}; Evidenz: {row.get('stage', 'nicht dokumentiert')}")
            comparators = row.get('classical_comparators', [])
            if not isinstance(comparators, list) or not comparators:
                classical.add('unknown')
            for item in comparators if isinstance(comparators, list) else []:
                if not isinstance(item, dict):
                    classical.add('unknown')
                    continue
                valid = item.get('evaluable') is True
                status = ('positive' if item.get('positive') is True else
                          'negative' if item.get('positive') is False else 'unknown') if valid else 'unknown'
                classical.add(status)
                details.append(f"Klassisch {item.get('criterion_id', '?')}: {status}")
        r, c = summarize(research), summarize(classical)
        comparison = ('Kriterienübereinstimmung' if r == c else 'Kriteriendiskordanz') if r in ('positive', 'negative') and c in ('positive', 'negative') else 'Keine eindeutige Gegenüberstellung'
        output.append(dict(endpoint_id=endpoint, name=rows[0].get('endpoint_name_de') or endpoint,
            research_status=r, classical_status=c, research_text=labels[r], classical_text=labels[c],
            comparison=comparison, details='\n'.join(sorted(set(details))),
            clinical_claim_allowed=False, probability=None))
    return output
