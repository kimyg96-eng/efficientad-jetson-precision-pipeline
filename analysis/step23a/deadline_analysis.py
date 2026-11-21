import argparse,csv,json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--capacity-source',type=Path);p.add_argument('--out',type=Path,default=Path('analysis'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
runs=[];groups=collections.defaultdict(list)
for f in sorted(a.source.glob('*/*/rate_*/round_*/summary.json')):
 s=json.loads(f.read_text());rows=list(csv.DictReader((f.parent/'images.csv').open()));v=np.array([float(r['arrival_to_decision_ms']) for r in rows]);assert len(v)==s['completed']
 for budget in range(20,221,5):
  n=int((v<=budget).sum());r={'category':s['category'],'engine':s['engine'],'rate_img_s':s['rate_img_s'],'round':s['round'],'deadline_ms':budget,'offered':s['offered'],'completed':s['completed'],'on_time':n,'on_time_pct_of_offered':100*n/s['offered'],'completed_cdf_pct':100*n/s['completed']};runs.append(r);groups[(s['category'],s['engine'],s['rate_img_s'],budget)].append(r)
assert len(runs)==5904
summary=[]
for (c,e,rate,b),rs in sorted(groups.items()):
 r={'category':c,'engine':e,'rate_img_s':rate,'deadline_ms':b,'rounds':len(rs)}
 for field in ['on_time_pct_of_offered','completed_cdf_pct']:
  vals=[x[field] for x in rs]
  for stat,fn in [('mean',np.mean),('min',np.min),('max',np.max)]:r[field+'_'+stat]=float(fn(vals))
 summary.append(r)
for name,rows in [('deadline_per_run.csv',runs),('deadline_summary.csv',summary)]:
 with (a.out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(2,4,figsize=(12,6),sharex=True,sharey=True)
for row,rate in enumerate([30,60]):
 for j,c in enumerate(['bottle','carpet','capsule','grid']):
  ax=axs[row,j]
  for e,label,color in [('default_tf32','TF32','#a34230'),('automatic_fp16','FP16','#007799')]:
   rs=[r for r in summary if r['category']==c and r['engine']==e and r['rate_img_s']==rate];x=[r['deadline_ms'] for r in rs];y=np.array([r['on_time_pct_of_offered_mean'] for r in rs]);lo=[r['on_time_pct_of_offered_min'] for r in rs];hi=[r['on_time_pct_of_offered_max'] for r in rs];ax.plot(x,y,label=label,color=color);ax.fill_between(x,lo,hi,color=color,alpha=.15)
  ax.set_title(f'{c.capitalize()} — {rate} img/s');ax.set_xticks([50,100,150,200]);ax.set_ylim(-2,102);ax.grid(alpha=.2)
axs[0,0].legend(fontsize=9,loc='lower right');fig.supxlabel('Exploratory decision deadline (ms)');fig.supylabel('On-time decisions (% of all offered arrivals)');fig.tight_layout();fig.savefig(a.out/'deadline_sweep.png',dpi=220);plt.close(fig)
# Finite-window Little identity: average number over window equals completion throughput * mean admitted sojourn.
# Boundary includes all read-start and finish events and completion drain; this is a sample-path identity, not p95 theory.
root=a.capacity_source or a.source.parent/'step22a_complete';little=[]
for f in root.glob('*/*/producers_4/capacity_*/round_*/summary.json'):
 s=json.loads(f.read_text());rows=list(csv.DictReader((f.parent/'images.csv').open()));duration=(s['end_ns']-s['start_ns'])/1e9;sojourn=np.array([(int(r['finished_ns'])-int(r['read_start_ns']))/1e9 for r in rows]);x=len(rows)/duration;L=float(sojourn.sum()/duration);W=float(sojourn.mean());little.append({'category':s['category'],'engine':s['engine'],'capacity':s['capacity'],'round':s['round'],'mean_number_from_read_start':L,'throughput_img_s':x,'mean_sojourn_ms':1000*W,'L_over_X_ms':1000*L/x,'identity_error_ms':1000*(L/x-W),'p95_ms':s['png_to_decision_ms_p95']})
assert len(little)==256
with (a.out/'little_sample_path.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(little[0]));w.writeheader();w.writerows(little)
(a.out/'README.md').write_text('''# STEP23A Exploratory deadline and queueing analysis

Run: python3 deadline_analysis.py /path/to/step21b_full_review --out analysis
Pass --capacity-source /path/to/extracted_STEP22A; otherwise the companion folder must be source.parent/step22a_complete. The sample-path table is required by this script. This script reads existing logs; it does not create new arrival-rate observations.

Deadline sweep 20–220 ms, step 5 ms, was added after reviewing fixed-rate results and is explicitly post hoc. Curves use all offered arrivals as denominator; dropped arrivals never contribute on-time decisions. Completed-only CDF is exported separately and must not be confused with delivery fraction. Means and shaded minima/maxima use six runs, not independent corpus images. A 100 ms sensitivity case existed before this post hoc sweep; it is not an application-derived deadline. No hard-real-time guarantee.

little_sample_path.csv computes finite-window read-start-to-decision occupancy from the sum of observed image sojourns, including completed drain. It checks an accounting identity Lbar = X Wbar. This is not an independent validation of queueing theory, does not set Lbar equal to configured capacity, and predicts no p95. Throughput uses the same finite host window as its occupancy integral. Waiting before read start is excluded, matching closed-loop latency.
''')
print('DEADLINE_ROWS',len(runs),'SUMMARY_ROWS',len(summary),'LITTLE_ROWS',len(little))
for c in ['bottle','carpet','capsule','grid']:
 for e in ['default_tf32','automatic_fp16']:
  print(c,e,[(b,round(next(r for r in summary if (r['category'],r['engine'],r['rate_img_s'],r['deadline_ms'])==(c,e,60,b))['on_time_pct_of_offered_mean'],2)) for b in [50,100,150,200]])
