import csv,json,collections
from pathlib import Path
import numpy as np
import argparse
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True);source=args.source;rows=[]
for p in sorted(source.glob('*/*/rate_*/round_*/summary.json')):
 s=json.loads(p.read_text())
 with (p.parent/'images.csv').open() as f:images=list(csv.DictReader(f))
 prep=np.array([(int(x['preprocess_end_ns'])-int(x['read_start_ns']))/1e6 for x in images]);wait=np.array([float(x['ready_wait_ms']) for x in images]);service=np.array([float(x['service_ms']) for x in images]);launch=np.array([(int(x['read_start_ns'])-int(x['release_ns']))/1e6 for x in images]);lag=np.array([float(x['release_lateness_ms']) for x in images]);total=np.array([float(x['arrival_to_decision_ms']) for x in images])
 assert np.allclose(lag+launch+prep+wait+service,total,atol=1e-6,rtol=0)
 span=(s['end_ns']-s['start_ns'])/1e9
 r={k:s[k] for k in ['category','engine','rate_img_s','round','offered','completed','drop_capacity','drop_release_late','drop_pct','delivered_during_arrival_window_img_s']}
 for name,v in [('dispatch',lag),('release_to_read',launch),('read_preparation',prep),('ready_wait',wait),('consumer_service',service),('arrival_to_decision',total)]:r[name+'_mean_ms']=float(v.mean())
 r['consumer_busy_pct_of_host_window']=float(service.sum()/1000/span*100)
 r['capacity_drop_pct']=100*s['drop_capacity']/s['offered'];r['scheduler_drop_pct']=100*s['drop_release_late']/s['offered'];rows.append(r)
def write(n,rs):
 with (root/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
write('stage_per_run.csv',rows);groups=collections.defaultdict(list)
for r in rows:groups[(r['category'],r['engine'],r['rate_img_s'])].append(r)
out=[]
for key,rs in sorted(groups.items()):
 d=dict(zip(['category','engine','rate_img_s'],key));d['rounds']=len(rs)
 for k in rs[0]:
  if k in d or k=='round':continue
  v=[r[k] for r in rs]
  for suffix,fn in [('mean',np.mean),('min',min),('max',max)]:d[k+'_'+suffix]=float(fn(v))
 out.append(d)
write('stage_summary.csv',out)
for c in ['carpet','capsule']:
 for rate in [55,70,80]:
  r=next(r for r in out if r['category']==c and r['engine']=='automatic_fp16' and r['rate_img_s']==rate)
  print(c,rate,{k:round(r[k],4) for k in ['drop_pct_mean','read_preparation_mean_ms_mean','release_to_read_mean_ms_mean','ready_wait_mean_ms_mean','consumer_service_mean_ms_mean','consumer_busy_pct_of_host_window_mean']})
(root/'README.md').write_text('''# STEP23D Drop categories and completed-image stage analysis

Analysis of the accepted STEP23B full session only: 240 arms and six rounds per condition. No new GPU measurements. Source: STEP23B_REVIEW(1).zip. Reproduction requires the extracted archive at the sibling step23b_full_review path, numpy and this script.

All offered = completed + capacity drops + scheduler-late drops. For completed images the scheduled-to-decision duration decomposes into release lateness + release-to-read + read/preparation + ready wait + consumer service. This identity is verified per image within 1e-6 ms. Means are averages of run means; min/max are observed across runs. Do not add component medians or p95 values.

Consumer busy fraction uses measured service duration divided by the finite host window through drain/join. It is not a CUDA utilization counter. Dropped images have no preparation/service trace, so completed-image component means cannot establish their counterfactual costs. Zero scheduler-late drops excludes that recorded rejection category; it does not eliminate all CPU scheduling effects on processing. Capacity rejection identifies the admission rule, not a unique hardware root cause. Comparing open-loop service with earlier closed-loop capacity does not isolate finite-capacity, arrival phasing, ordered consumption or preparation variability.
''')
