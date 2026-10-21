"""Independently verify complete archives and count each attempted unit node once."""
import argparse,collections,hashlib,json,re,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--continuation',required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();a.out_dir.mkdir(exist_ok=True,parents=True)
archives=[tarfile.open(a.initial),tarfile.open(a.continuation)]
selected=set();started={};reports=collections.defaultdict(list);blocked=set();errors={};windows=[];verified=0

def clean(s):
 s=s.replace('/home/zhen/dezhenlu/mlm-20261020','<WORK_DIR>').replace('/home/zhen','<HOME>').replace('/tmp/pytest-of-zhen','<PYTEST_TMP>')
 return s

def bucket(s):
 low=s.lower()
 if 'experimentalnotenabled' in low:return 'experimental_flag'
 if any(t in low for t in ['world_size','world size','invalid device ordinal','not divisible by','cuda device count']):return 'topology_or_world_size'
 if any(t in low for t in ['no module named','importerror','cannot import name','attributeerror']):return 'dependency_or_api'
 if any(t in low for t in ['filenotfounderror','offline mode','cached files','/opt/data','/opt/megatron','no such file']):return 'missing_path_or_asset'
 if any(t in low for t in ['outofmemory','out of memory','outofresources']):return 'resource_limit'
 return 'other_failure'

for archive in archives:
 names={m.name:m for m in archive.getmembers() if m.isfile()}
 for name in names:
  if not name.endswith('/SHA256.json'):continue
  manifest=json.load(archive.extractfile(name));root=name.rsplit('/',1)[0]
  for relative,expected in manifest.items():
   member=root+'/'+relative;stream=archive.extractfile(member);h=hashlib.sha256()
   while chunk:=stream.read(1<<20):h.update(chunk)
   assert h.hexdigest()==expected,member
   verified+=1
 journals=sorted(n for n in names if n.endswith('/outcomes.jsonl') or re.search(r'/rounds/round-\d+\.jsonl$',n))
 for name in journals:
  events_started=0;finished=False
  for line in archive.extractfile(name):
   e=json.loads(line);event=e['event'];n=e.get('nodeid')
   if event=='collection':selected.update(e['nodeids'])
   elif event=='start':
    assert n not in started,(n,'duplicate attempt');started[n]=name;events_started+=1
   elif event=='report':
    detail=e.get('detail') or '';reports[n].append({'phase':e['phase'],'outcome':e['outcome'],'wasxfail':e.get('wasxfail'),'failure_text_bucket':bucket(detail),'detail':clean(detail[-1200:])})
   elif event=='resume_selection':blocked.update(e['blocked_unexecuted'])
   elif event=='collection_error':errors[n]={'failure_text_bucket':bucket(e['detail']),'detail':clean(e['detail'][-1200:])}
   elif event=='finish':finished=True
  windows.append({'journal':name,'started':events_started,'finished':finished})
ledger=[]
for n in sorted(selected):
 rs=reports[n];status='blocked_after_file_timeout' if n in blocked else 'not_attempted';detail=None;kind=None
 if n in started:
  bad=[x for x in rs if x['outcome']=='failed']
  if bad:status='failed';detail=bad[0]['detail'];kind=bad[0]['failure_text_bucket']
  elif not any(x['phase']=='teardown' for x in rs):status='interrupted';detail='Started without teardown before process termination.'
  elif any(x['outcome']=='skipped' for x in rs):
   hit=next(x for x in rs if x['outcome']=='skipped');status='xfailed' if hit['wasxfail'] else 'skipped';detail=hit['detail']
  elif any(x['phase']=='call' and x['outcome']=='passed' for x in rs):
   hit=next(x for x in rs if x['phase']=='call');status='xpassed' if hit['wasxfail'] else 'passed'
  else:status='unknown'
 ledger.append({'nodeid':n,'status':status,'window':started.get(n),'failure_text_bucket':kind,'detail':detail})
summary={'source_commit':'67413f611ffb1e7d7d29cd2a61f520cffe79d995','world_size':1,'selected_non_internal':len(selected),'attempted_unique':len(started),'counts':dict(collections.Counter(x['status'] for x in ledger)),'failure_text_buckets':dict(collections.Counter(x['failure_text_bucket'] for x in ledger if x['status']=='failed')),'collection_error_modules':len(errors),'collection_errors':errors,'windows':windows,'verified_original_files':verified,'archives':{Path(path).name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [a.initial,a.continuation]},'scope':'Native aarch64 GH200 world=1 compatibility inventory, not 8-GPU CI parity. Timed-out file tails remain unexecuted. Text buckets are heuristic labels; original full outputs remain in the verified archives.'}
(a.out_dir/'aarch64-unit-inventory-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.out_dir/'aarch64-unit-inventory-ledger.jsonl').open('w') as stream:
 for row in ledger:stream.write(json.dumps(row)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['collection_errors','windows']},indent=2))
