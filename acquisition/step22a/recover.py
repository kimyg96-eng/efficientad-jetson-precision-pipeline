"""Read-only source verification and a new complete evidence tree; never edits source."""
import argparse,csv,hashlib,json,shutil,sys
from pathlib import Path
from conditions import ordered_arms
PIN='bd4b32dafac624f82666c7b287aa809e4f80f693919322f0045d8ae1b54548fd'
CATS=('bottle','carpet','capsule','grid')
def key(s):return s['category'],s['engine'],int(s['producers']),int(s['capacity']),int(s['round'])
def relative(k):
 c,e,p,b,r=k;return Path(c)/e/f'producers_{p}'/f'capacity_{b}'/f'round_{r}'
def all_order():return [(c,e,p,b,r) for c in CATS for r in range(1,9) for e,p,b in ordered_arms('full',r)]
def select_plan(available):
 expected=all_order();assert set(available)<=set(expected)
 missing=set(expected)-set(available)
 # Reacquire complete engine pairs when either engine is missing; no cross-session pair.
 pair_keys={(c,p,b,r) for c,e,p,b,r in missing}
 plan=[k for k in expected if (k[0],k[2],k[3],k[4]) in pair_keys]
 retained=[k for k in expected if k in available and k not in plan]
 return plan,retained

def prepare(source,out,copy=True):
 source=source.resolve(strict=True);out=out.resolve();manifest=source/'SHA256SUMS.txt'
 if hashlib.sha256(manifest.read_bytes()).hexdigest()!=PIN:raise ValueError('Source manifest differs from reviewed interrupted run')
 files=[]
 for line in manifest.read_text().splitlines():
  digest,name=line.split(maxsplit=1);path=(source/name.lstrip('*')).resolve(strict=True)
  if not path.is_relative_to(source) or not path.is_file():raise ValueError('Unexpected source path')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError(f'Source hash mismatch: {name}')
  files.append(name)
 available={}
 for p in source.glob('*/*/producers_*/capacity_*/round_*/summary.json'):
  s=json.loads(p.read_text());k=key(s)
  if k in available or p.parent.relative_to(source)!=relative(k):raise ValueError('Duplicate or misplaced result')
  if s['replay']!='PASS' or s['duration_s']<60 or s['max_power_gap_ms']>150 or s['max_inflight']>s['capacity']:raise ValueError('Invalid accepted source run')
  available[k]=s
 with (source/'per_run.csv').open(newline='') as f:csvrows=list(csv.DictReader(f))
 if len(available)!=493 or len(csvrows)!=493 or {key(s) for s in csvrows}!=set(available):raise ValueError('Unexpected source run inventory')
 plan,retained=select_plan(available)
 if len(plan)!=20 or len(retained)!=492:raise ValueError('Unexpected resume plan')
 superseded=[k for k in available if k not in retained]
 info={'source_directory':str(source),'source_manifest_sha256':PIN,'source_hashes_verified':len(files),'original_completed':493,'retained':492,'reacquire':20,'expected_combined':512,'superseded_completed_pairs':[list(k) for k in superseded],'resume_keys':[list(k) for k in plan],'policy':'both engines reacquired if either arm absent; old source immutable; 150 ms power-gap limit unchanged; no automatic retries','session_caveat':'original 492 arms and resumed 20 arms span sessions; pair-local contrasts retained, session/environment changes must be reported'}
 if copy:
  if any(out.glob('*/default_tf32/producers_*')) or (out/'retained_runs.json').exists():raise ValueError('Output already contains results')
  for k in retained:shutil.copytree(source/relative(k),out/relative(k))
  (out/'retained_runs.json').write_text(json.dumps([available[k] for k in retained],indent=2)+'\n')
  (out/'recovery_plan.json').write_text(json.dumps(info,indent=2)+'\n')
  with (out/'acquisition_sessions.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['category','engine','producers','capacity','round','session','relative_directory'])
   for k in all_order():w.writerow([*k,'retained_original' if k in retained else 'reacquired_recovery',relative(k).as_posix()])
  shutil.copy2(manifest,out/'STEP22A_SOURCE_SHA256SUMS.txt')
  shutil.copy2(source/'protocol.json',out/'original_protocol.json')
  shutil.copy2(source/'clock_bounds_restore.log',out/'source_clock_bounds_restore.log')
 print(json.dumps(info,indent=2),flush=True);return info
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args();prepare(a.source,a.out,not a.check_only)
