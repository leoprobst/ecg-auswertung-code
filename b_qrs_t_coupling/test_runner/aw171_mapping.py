"""Operativer Parser-Kandidat: Klauseln und Negation; reale Validierung noch offen."""
import re
from werkzeuge.b_test.aw171_kernels import require

TOKEN=re.compile(r'\b(?:paced|pacemaker|v[- ]paced|a[- ]paced|av[- ]paced|ventricular pacing|atrial pacing|pacing)\b',re.I)
NEG=re.compile(r'\b(?:no|not|without|absence of|kein|keine|nicht)\b',re.I)
HIST=re.compile(r'\b(?:history|historical|previous|prior|formerly|anamnes|früher)\w*\b',re.I)
UNCERTAIN=re.compile(r'\b(?:possible|possibly|suspected|questionable|cannot exclude|rule out|consider|vermutet)\b|\?',re.I)

def pacing(text,usable=True):
    if not usable or text is None or not str(text).strip(): return 'unknown'
    clauses=re.split(r'[.;\n]|\bbut\b',str(text).lower())
    states=[]
    for clause in clauses:
        if not TOKEN.search(clause): continue
        if UNCERTAIN.search(clause): states.append('suspected')
        elif HIST.search(clause): states.append('historical')
        elif NEG.search(clause): states.append('negated')
        else: states.append('positive')
    if not states:return 'no_mention'
    if 'positive' in states:
        # Keine automatische positive Entscheidung bei widersprüchlichen Klauseln.
        if 'negated' in states or 'suspected' in states:return 'ambiguous'
        return 'positive'
    if len(set(states))>1:return 'ambiguous'
    return states[0]

def medication(positive,coverage):
    return 'exposed' if positive is True else ('observed_no_record' if coverage is True else 'unknown')

def measurement(value,mapping):
    if value is None: return 'unbekannt'
    require(value in mapping and value not in ['unmapped'],'Messart_neuer_Quellwert')
    return mapping[value]

def setting(ed,ip,ambiguous=False,complete=True):
    if ambiguous or not complete:return 'unknown'
    if ed and ip:return 'both'
    if ed:return 'ED'
    if ip:return 'inpatient'
    return 'outside'

def icd(code,version,groups):
    require(version in [9,10],'ICD_Version')
    value=str(code).strip().upper().replace('.','')
    return [g['group'] for g in groups if any(value.startswith(prefix) for prefix in g['icd'+str(version)])]
