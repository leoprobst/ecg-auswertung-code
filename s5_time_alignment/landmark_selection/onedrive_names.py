"""Conservative <cloud folder> names for copies; scientific identifiers stay intact."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path

_DEVICES=re.compile(r'^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])$',re.I)
_SAFE=re.compile(r'^[A-Za-z0-9_.=\-]+$')


def compatible_component(name: str) -> bool:
    return bool(name and len(name)<=200 and _SAFE.fullmatch(name)
        and name==name.strip(' .') and '_vti_' not in name.casefold()
        and name.casefold() not in ('.lock','desktop.ini','forms')
        and not _DEVICES.fullmatch(name.split('.')[0]))


def safe_component(name: str) -> str:
    if compatible_component(name):return name
    cleaned=re.sub(r'[^A-Za-z0-9_.=\-]','_',name).strip(' .')
    cleaned=re.sub('_vti_','_voltage_time_integral_',cleaned,flags=re.I)
    if not cleaned:cleaned='file'
    if cleaned.casefold() in ('.lock','desktop.ini','forms') or _DEVICES.fullmatch(cleaned.split('.')[0]):
        cleaned='file_'+cleaned.lstrip('.')
    suffix=Path(cleaned).suffix
    if len(suffix)>20:suffix=''
    stem=cleaned[:-len(suffix)] if suffix else cleaned
    digest=hashlib.sha256(name.encode('utf-8')).hexdigest()[:12]
    result=stem[:200-len(suffix)-14]+'__'+digest+suffix
    assert compatible_component(result)
    return result


def safe_relative_path(path: Path) -> Path:
    if path.is_absolute() or any(p in ('..','.') for p in path.parts):
        raise ValueError('Expected a contained relative path')
    return Path(*(safe_component(p) for p in path.parts))


def planned_destinations(paths,source_root: Path,destination_root: Path):
    result={};seen={}
    for source in paths:
        relative=source.relative_to(source_root)
        target=destination_root/safe_relative_path(relative)
        if len(str(target))>400:
            raise ValueError('<cloud folder> path is too long: '+str(target))
        key=str(target).casefold()
        if key in seen and str(seen[key])!=str(source):
            raise ValueError('Destination name collision: '+str(target))
        seen[key]=source;result[source]=target
    return result
