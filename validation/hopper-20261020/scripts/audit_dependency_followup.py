"""Audit the separate dependency-repair run without modifying the original inventory."""
import argparse,collections,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',required=True);p.add_argument('--scope',default='Separate world=1 rerun after adding isolated datasets/FastAPI/uvicorn dependencies and selecting --experimental --import-mode=importlib. Does not overwrite initial failures or count as 8-GPU CI.');a=p.parse_args()
manifest=json.loads((a.directory/'SHA256.json').read_text());assert all(hashlib.sha256((a.directory/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
def clean(s):
 s=(s or '').replace('/home/zhen/dezhenlu/mlm-20261020','<WORK_DIR>').replace('/home/zhen','<HOME>').replace('/tmp/pytest-of-zhen','<PYTEST_TMP>')
 return re.sub(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])','<IP>',s)
records=[];ledger=[]
for case in json.loads((a.directory/'cases.json').read_text()):
 f=a.directory/(case['name']+'.jsonl');events=[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
 selected=set();started=set();reports=collections.defaultdict(list);errors=[]
 for e in events:
  if e['event']=='collection':selected.update(e['nodeids'])
  elif e['event']=='start':started.add(e['nodeid'])
  elif e['event']=='report':reports[e['nodeid']].append(e)
  elif e['event']=='collection_error':errors.append({'nodeid':e.get('nodeid'),'detail':clean(e.get('detail'))})
 local=[]
 for node in sorted(selected):
  rs=reports[node];bad=[r for r in rs if r['outcome']=='failed'];skips=[r for r in rs if r['outcome']=='skipped'];passed=[r for r in rs if r['phase']=='call' and r['outcome']=='passed'];detail=None
  if bad:status='failed';detail=bad[0].get('detail')
  elif node not in started:status='not_attempted'
  elif not any(r['phase']=='teardown' for r in rs):status='interrupted'
  elif skips:status='xfailed' if skips[0].get('wasxfail') else 'skipped';detail=skips[0].get('detail')
  elif passed:status='xpassed' if passed[0].get('wasxfail') else 'passed'
  else:status='unknown'
  local.append({'nodeid':node,'case':case['name'],'status':status,'detail':clean(detail)})
 ledger.extend(local);records.append({'case':case['name'],'target':case['argv'][-1],'driver_exit_code':case['exit_code'],'collected':len(selected),'attempted':len(started),'counts':dict(collections.Counter(r['status'] for r in local)),'collection_errors':errors,'journal_finish':any(e['event']=='finish' for e in events)})
result={'verified_files':len(manifest),'executed_files_or_targets':len(records),'counts':dict(collections.Counter(r['status'] for r in ledger)),'records':records,'ledger':ledger,'scope':a.scope}
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','ledger']},indent=2))
