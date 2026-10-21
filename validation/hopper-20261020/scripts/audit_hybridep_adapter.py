"""Verify the complete adapter archive and summarize every expected rank record."""
import argparse,collections,hashlib,json,math,statistics,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--out',required=True);a=p.parse_args()
with tarfile.open(a.archive) as t:
 manifest=json.load(t.extractfile('adapter-evidence/SHA256.json'))
 for name,expected in manifest.items():
  h=hashlib.sha256()
  with t.extractfile('adapter-evidence/'+name) as f:
   while chunk:=f.read(1<<20):h.update(chunk)
  assert h.hexdigest()==expected,name
 rows=[]
 for member in t.getmembers():
  if member.name.startswith('adapter-evidence/results/') and member.name.endswith(('/rank0.jsonl','/rank1.jsonl')):
   rows.extend(json.loads(line) for line in t.extractfile(member))
 groups=collections.defaultdict(list)
 for r in rows:groups[(r['mode'],r['workload'],r['backend'],r['rank'])].append(r)
 for key,rs in groups.items():
  assert sorted(r['step'] for r in rs)==list(range(6)),key
 failed=[{k:r[k] for k in ['mode','workload','backend','step','rank','errors_vs_hybridep']} for r in rows if not r['numerical_pass']]
 paired={}
 for r in rows:
  key=(r['mode'],r['workload'],r['step'],r['rank'])
  if r['backend']=='hybridep':paired[key]=r
 for r in rows:
  if r['backend']=='hybridep_virtual_experts':
   ref=paired[(r['mode'],r['workload'],r['step'],r['rank'])]
   assert r['input_sha256']==ref['input_sha256'] and r['routes_sha256']==ref['routes_sha256']
   assert r['logical_expert_counts']==ref['logical_expert_counts']
 summaries=[]
 for key,rs in groups.items():
  if key[-1]:continue
  summaries.append({'mode':key[0],'workload':key[1],'backend':key[2], 'worst_rank_ms_mean':statistics.mean(r['worst_rank_ms'] for r in rs),'rank_time_max_over_mean_mean':statistics.mean(r['rank_time_max_over_mean'] for r in rs),'logical_rank_max_over_mean_per_step':[r['logical_rank_max_over_mean'] for r in rs]})
 result={'archive_sha256':hashlib.sha256(Path(a.archive).read_bytes()).hexdigest(),'verified_files':len(manifest),'expected_rank_records':336,'observed_rank_records':len(rows),'numerical_gate_pass':len(rows)==336 and not failed,'failed_records':failed,'summaries':summaries,'scope':'BF16 finite MoELayer, EP2 TP1 ETP1, seven W workloads x six steps x two backends x two ranks x two modes. Fixed order timings are descriptive, not speedup qualification.'}
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['summaries','failed_records']},indent=2))
