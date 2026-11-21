import csv,json,hashlib,sys,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root.parent/'step23b'))
from power_monitor import integrate
protocol=json.loads((root/'protocol.json').read_text());assert protocol['mode']=='full'
expected={(c,e,r,k) for c in ['bottle','carpet','capsule','grid'] for e in ['default_tf32','automatic_fp16'] for r in [35,45,55,70,80] for k in range(1,7)}
manifest=[]
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
 digest,name=line.split('  ',1);p=root/name;assert p.is_file() and not p.is_symlink();assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;manifest.append(name)
seen=set();groups=collections.defaultdict(list);curves=[];total=offered=errors=0;gapmax=energyerr=scoreerr=0.;tele=collections.defaultdict(list);runs=[]
for p in sorted(root.glob('*/*/rate_*/round_*/summary.json')):
 s=json.loads(p.read_text());key=(s['category'],s['engine'],s['rate_img_s'],s['round']);assert key not in seen;seen.add(key)
 def read(n):
  with (p.parent/n).open() as f:return list(csv.DictReader(f))
 rows,a,power=read('images.csv'),read('arrivals.csv'),read('power.csv')
 assert len(rows)==s['completed'] and len(a)==s['offered']==s['rate_img_s']*60
 assert len(rows)+s['drop_capacity']+s['drop_release_late']==len(a)
 assert [int(x['sequence']) for x in rows]==[int(x['sequence']) for x in a if x['status']=='accepted']
 assert sum(x['status']!='accepted' for x in a)==s['drop_capacity']+s['drop_release_late']
 for x in rows:
  err=abs(float(x['score'])-float(x['expected_score']));scoreerr=max(scoreerr,err);assert err<=1e-6
  assert np.isfinite(float(x['score'])) and int(x['decision'])==int(float(x['score'])>float(x['threshold']))
 values=np.array([float(x['arrival_to_decision_ms']) for x in rows]);assert np.isfinite(values).all() and np.all(values>=0)
 assert abs(np.percentile(values,95)-s['arrival_to_decision_ms_p95'])<1e-9
 for b in [50,75,100,125,150,175,200]:assert int((values<=b).sum())==s[f'on_time_{b}ms']
 for b in range(20,221,5):curves.append({'category':key[0],'engine':key[1],'rate_img_s':key[2],'round':key[3],'deadline_ms':b,'on_time_pct_of_offered':100*int((values<=b).sum())/s['offered']})
 en,gap=integrate(power,s['start_ns'],s['end_ns']);energyerr=max(energyerr,abs(sum(en.values())-s['three_rail_energy_j']));gapmax=max(gapmax,gap)
 assert abs(100*(s['offered']-s['completed'])/s['offered']-s['drop_pct'])<1e-9
 errors+=len(json.loads((p.parent/'sensor_read_errors.json').read_text()));total+=len(rows);offered+=len(a)
 for x in read('telemetry.csv'):
  for k,v in x.items():
   if k!='monotonic_ns' and v:tele[k].append(int(v))
 runs.append(s);groups[key[:3]].append(s)
assert seen==expected and len(manifest)==1476
summary=[]
fields=['drop_pct','arrival_to_decision_ms_p95','on_time_100ms_pct_of_offered','completed_with_drain_img_s','mj_per_completed','mean_three_rail_w']
for key,rs in sorted(groups.items()):
 assert len(rs)==6
 row={'category':key[0],'engine':key[1],'rate_img_s':key[2],'rounds':6,'all_rounds_zero_drop':all(s['completed']==s['offered'] for s in rs)}
 for f in fields:
  v=[s[f] for s in rs]
  for suffix,fn in [('mean',np.mean),('min',min),('max',max)]:row[f+'_'+suffix]=float(fn(v))
 summary.append(row)
def write(n,rows):
 with (out/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('condition_summary.csv',summary);write('deadline_per_run.csv',curves)
fig,axs=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
for ax,c in zip(axs.flat,['bottle','carpet','capsule','grid']):
 for e,label,color in [('default_tf32','Default TF32','#a34230'),('automatic_fp16','Automatic FP16','#007799')]:
  rs=[s for s in summary if s['category']==c and s['engine']==e];x=[s['rate_img_s'] for s in rs];v=np.array([s['drop_pct_mean'] for s in rs]);lo=np.array([s['drop_pct_min'] for s in rs]);hi=np.array([s['drop_pct_max'] for s in rs]);ax.errorbar(x,v,yerr=[v-lo,hi-v],marker='o',capsize=3,color=color,label=label)
 ax.set_title(c.capitalize());ax.set_xticks([35,45,55,70,80]);ax.grid(alpha=.2);ax.set_ylim(-2,63)
axs[0,0].legend(fontsize=9);fig.supxlabel('Offered arrival rate (images/s)');fig.supylabel('Dropped arrivals (%)');fig.tight_layout();fig.savefig(out/'dense_arrival_drops.png',dpi=220);plt.close(fig)
audit={'status':'PASS','runs':len(seen),'hash_files':len(manifest),'offered':offered,'completed':total,'sensor_read_errors':errors,'max_score_replay_error':scoreerr,'max_energy_reintegration_error_j':energyerr,'max_power_gap_ms':gapmax,'telemetry_ranges':{k:[min(v),max(v)] for k,v in tele.items()},'restoration':(root/'clock_bounds_restore.log').read_text()}
(out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
lines=['# STEP23C Dense arrival full experiment review','', '240 prespecified arms completed in a fresh full session; the interrupted 35-arm session and pilot are excluded. Six rounds per condition, 60 seconds offered arrivals, P4 and total capacity five. Do not pool condition means with earlier 15/30/60 sessions.','', '|Category|TF32 drop at 35|FP16 largest tested zero-drop rate|FP16 drop at 70|FP16 drop at 80|','|---|---:|---:|---:|---:|']
for c in ['bottle','carpet','capsule','grid']:
 get=lambda e,r:next(s for s in summary if s['category']==c and s['engine']==e and s['rate_img_s']==r)
 good=[s['rate_img_s'] for s in summary if s['category']==c and s['engine']=='automatic_fp16' and s['all_rounds_zero_drop']]
 lines.append(f"|{c}|{get('default_tf32',35)['drop_pct_mean']:.3f}%|{max(good)}|{get('automatic_fp16',70)['drop_pct_mean']:.3f}%|{get('automatic_fp16',80)['drop_pct_mean']:.3f}%|")
lines+=['','Largest tested zero-drop rate requires zero drops in every round. Bottle/Grid 80 is an observed lower bound on attainable zero-drop rate, not a measured upper limit. Carpet/Capsule show no drops through 55 and drops at 70; the onset is not exactly identified. The earlier 60-rate zero-drop finding is from a separate session. TF32 dropped at every rate in this new grid; the earlier 30-rate result provides separate contextual evidence, not a pooled boundary estimate.','', 'Deadline curves count all offered inputs including drops. Summary ranges are observed run minima/maxima, not confidence intervals. Preprocessing, schedule, finite capacity, ordered consumption and clock policy jointly determine delivery. No camera or hard-real-time guarantee.','',f"Audit: {len(manifest)} hashes; {total} decisions; sensor errors {errors}; maximum gap {gapmax:.6f} ms; energy replay error {energyerr:.12g} J. Clock restoration passed. Raw archive: STEP23B_REVIEW(1).zip. Reproduce: python analyze.py <extracted_full_review> <output>; trusted power_monitor.py is included in this analysis folder for portable use."]
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(audit,indent=2));print('\n'.join(lines[4:11]))
