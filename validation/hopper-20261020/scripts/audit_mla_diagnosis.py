import argparse,hashlib,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',required=True);a=p.parse_args();manifest=json.loads((a.directory/'SHA256.json').read_text());assert all(hashlib.sha256((a.directory/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
records={f.name:json.loads(f.read_text()) for f in a.directory.glob('mla-*.json') if '.partial.' not in f.name}
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
by_mode=collections.defaultdict(list)
for name,d in records.items():by_mode[d['mode']].append((name,d))
summary={}
for mode,pairs in by_mode.items():
 fields={'initial_parameters':lambda d:d['initial']['parameters'],'input_tokens':lambda d:d['initial']['tokens'],'loss_sequence':lambda d:[s['loss_digest'] for s in d['steps']],'final_parameters':lambda d:d['parameter_digests'],'final_gradients':lambda d:d['gradient_digests'],'choices':lambda d:d['choice_digest']}
 variants={k:collections.defaultdict(list) for k in fields}
 for name,d in pairs:
  for k,fn in fields.items():variants[k][digest(fn(d))].append(name)
 summary[mode]={'runs':len(pairs),'variant_counts':{k:len(v) for k,v in variants.items()},'variants':variants,'loss_values':{n:[s['loss'] for s in d['steps']] for n,d in pairs}}
result={'verified_files':len(manifest),'runs':len(records),'summary':summary,'scope':'Independent process diagnosis with explicit initial parameter/input hashes; does not replace the original failed qualification or performance gates.'};Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({m:{'runs':v['runs'],'variant_counts':v['variant_counts']} for m,v in summary.items()},indent=2))
