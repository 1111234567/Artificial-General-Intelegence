"""Wait for all declared evidence, then shard independent raw audits by seed."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import time
from experiments.wait_position import eval_position as e,eval_report

def worker(seed):
    target=e.OUT/f'audit_seed_{seed:03d}.json'
    if target.exists():return e.read(target)
    with (e.OUT/f'audit_seed_{seed:03d}.log').open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u','-m','experiments.wait_position.eval_position','audit','--seed',str(seed),'--complete'],cwd=e.ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,('audit seed failed',seed)
    return e.read(target)

def finalize():
    plan=e.checked()
    wanted=[e.result_path(s,a,t,q) for s in range(3) for a in e.ARMS for t in e.SNAPSHOTS for q in e.scenarios(t)]
    assert len(wanted)==210
    while not all(p.exists() for p in wanted):time.sleep(5)
    with ThreadPoolExecutor(max_workers=3) as pool:shards=list(pool.map(worker,range(3)))
    for seed,shard in enumerate(shards):
        assert shard['verified'] and shard['complete'] and shard['seed']==seed and shard['records']==70
        assert shard['protocol_hash']==plan['protocol_hash']
    e.write(e.OUT/'audit.json',dict(format=e.FORMAT,protocol_hash=plan['protocol_hash'],verified=True,complete=True,records=210,
        matched_external_groups=9,unique_evidence_forks=sum(s['unique_evidence_forks'] for s in shards),checks=shards[0]['checks'],
        shard_reports={str(e.OUT/f'audit_seed_{s:03d}.json'):e.sha(e.OUT/f'audit_seed_{s:03d}.json') for s in range(3)}))
    eval_report.run()
    value=dict(format=e.FORMAT,complete=True,records=210,protocol_hash=plan['protocol_hash'],source_sha256=e.sha(__file__),
        artifacts={str(e.OUT/name):e.sha(e.OUT/name) for name in ('audit.json','summary.json','timing-summary.json','report-binding.json','curves.png','report.md')})
    e.write(e.OUT/'completion.json',value);print(json.dumps(dict(event='wait_position_evaluation_complete',records=210)),flush=True)
    return value
if __name__=='__main__':finalize()
