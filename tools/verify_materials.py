#!/usr/bin/env python3
"""Read-only SHA-256 checks for the source snapshot and optional release assets (Python 3.11+)."""
import argparse,csv,hashlib,json,re,sys
from pathlib import Path

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(root,assets=None):
 root=root.resolve(strict=True);errors=[];source_count=asset_count=0
 for line in (root/'SHA256SUMS.txt').read_text().splitlines():
  m=re.fullmatch(r'([0-9a-f]{64})  (.+)',line)
  if not m:raise ValueError('Malformed source hash record')
  relative=Path(m[2])
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe source hash path')
  p=root/relative
  if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root):errors.append({'file':m[2],'reason':'missing_nonregular_or_outside_root'});continue
  if sha(p)!=m[1]:errors.append({'file':m[2],'reason':'source_hash_mismatch'})
  else:source_count+=1
 with (root/'RAW_DATA_MANIFEST.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 names=[r['archive'] for r in rows]
 if len(set(names))!=len(names):raise ValueError('Duplicate release asset names')
 if assets is not None:
  assets=assets.resolve(strict=True)
  for row in rows:
   n=row['archive']
   if Path(n).name!=n or not n:raise ValueError('Unsafe asset name')
   p=assets/n
   if not p.is_file():errors.append({'file':n,'reason':'missing_asset'});continue
   if p.stat().st_size!=int(row['bytes']):errors.append({'file':n,'reason':'asset_size_mismatch'});continue
   if sha(p)!=row['sha256']:errors.append({'file':n,'reason':'asset_hash_mismatch'});continue
   asset_count+=1
 result={'status':'PASS' if not errors else 'FAIL','source_files_matched':source_count,'expected_release_assets':len(rows),'asset_check':'NOT_REQUESTED' if assets is None else 'REQUESTED','assets_matched':asset_count,'errors':errors}
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--assets',type=Path);a=p.parse_args()
 result=verify(a.root,a.assets);print(json.dumps(result,indent=2));sys.exit(0 if result['status']=='PASS' else 1)
