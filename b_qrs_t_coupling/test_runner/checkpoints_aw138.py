"""Checks existing checkpoints of the extraction stage before resuming; technical aggregates only."""
import hashlib
import json
from datetime import datetime, timezone
from . import extraktion_aw138 as a
from werkzeuge.nachrechnung.resources import guard, Budget

def main():
    start = guard()
    from pathlib import Path
    source = Path(__file__)
    snapshot = a.OUT/'executed_source'/a.sha(source)/source.name
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes(source.read_bytes())
    with Budget(a.OUT) as budget:
        tasks = a.readp(a.OUT/'marker_tasks_private.parquet', ['subject_id','study_id','split'])
        if tasks.study_id.duplicated().any() or not tasks.split.eq('locked_test').all():
            raise ValueError('Task identity or partition')
        expected = tasks.set_index('study_id').subject_id.to_dict()
        request = a.sha(a.OUT/'marker_tasks_private.parquet')
        options = hashlib.sha256(json.dumps(a.LIMIT_OPTIONS, sort_keys=True).encode()).hexdigest()
        full_options = hashlib.sha256(json.dumps(a.OPTIONS, sort_keys=True).encode()).hexdigest()
        reused = 0
        inventory = []
        prior_inventory = a.OUT/'resume_checkpoint_hashes_private.json'
        prior_hashes = {}
        if prior_inventory.exists():
            prior_hashes = {v['file']:v['sha256'] for v in a.readj(prior_inventory)}
        for p in sorted((a.OUT/'marker_records_v2').glob('*.json')):
            obj = json.loads(p.read_text(encoding='utf8'))
            r = obj['result']
            study = int(p.stem)
            if (obj['request_sha256'] != request or obj['options_sha256'] != options
                or obj.get('adapter_sha256') is not None or r['study_id'] != study
                or study not in expected or r['subject_id'] != expected[study]
                or r['split'] != 'locked_test' or not r.get('extraction_status')
                or any(k not in r for k in a.FIELDS)):
                raise ValueError('Checkpoint schema or binding')
            prior = obj.get('reused_from')
            if prior and a.sha(prior['path']) != prior['sha256']:
                raise ValueError('Parent checkpoint changed')
            if prior:
                parent = a.readj(prior['path'])
                if (parent['request_sha256'] != request or parent['options_sha256'] != full_options
                    or parent.get('adapter_sha256') is not None or parent['result'] != r
                    or prior['subset_equivalence_attestation_sha256'] != a.sha(a.OUT/'limited_equivalence_attestation.json')):
                    raise ValueError('Parent request, options or result differ')
                reused += 1
            digest = a.sha(p)
            if p.name in prior_hashes and digest != prior_hashes[p.name]:
                raise ValueError('Previously attested checkpoint changed')
            inventory.append({'file':p.name,'sha256':digest,'bytes':p.stat().st_size})
        if not set(prior_hashes).issubset({v['file'] for v in inventory}):
            raise ValueError('Previously attested checkpoint missing')
        final = len(inventory) == len(tasks)
        prefix = 'final_checkpoint' if final else 'resume_checkpoint'
        a.save(prefix+'_hashes_private.json', inventory)
        a.save(prefix+'_attestation.json', {
            'passed':True,'utc':datetime.now(timezone.utc).isoformat(),
            'checked_records':len(inventory),'requested_records':len(tasks),
            'remaining_records':len(tasks)-len(inventory),
            'request_sha256':request,'options_sha256':options,
            'inventory_sha256':a.sha(a.OUT/(prefix+'_hashes_private.json')),
            'prior_attested_records_unchanged':len(prior_hashes),
            'full_parent_records_checked':reused,
            'schema_identity_partition_and_parent_hash_checked':True,
            'boundary':'Previous individual file hashes were not archived separately; current hash inventory plus request/options/parent binding, no retroactive proof of history.',
            'script_sha256':a.sha(__file__)})
    a.save('resources_'+prefix+'_checkpoint_audit.json', {'start':start,**budget.summary(),'script_sha256':a.sha(__file__)})
    print(json.dumps(a.readj(a.OUT/(prefix+'_attestation.json'))), flush=True)

if __name__ == '__main__':
    main()
