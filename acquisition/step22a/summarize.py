"""Validate complete acquisition and summarize runs; no image-level pseudoreplication."""
import argparse,csv,json,collections
from pathlib import Path
import numpy as np
from conditions import cells
from power_monitor import integrate
p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();root=a.out
protocol=json.loads((root/'protocol.json').read_text());mode=protocol['mode'];rounds=protocol['rounds'];cats=protocol['categories'];expected_cells=cells(mode)
def read(path):return list(csv.DictReader(path.open(newline='')))
def write(name,rs):
 with (root/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
runs=read(root/'per_run.csv');expected={(c,e,w,b,r) for c in cats for e in ('default_tf32','automatic_fp16') for w,b in expected_cells for r in range(1,rounds+1)}
actual={(s['category'],s['engine'],int(s['producers']),int(s['capacity']),int(s['round'])) for s in runs}
assert len(runs)==len(expected) and actual==expected
max_energy_error=0;total_images=0;error_count=0
for s in runs:
 directory=root/s['category']/s['engine']/('producers_'+s['producers'])/('capacity_'+s['capacity'])/('round_'+s['round'])
 images=read(directory/'images.csv');assert len(images)==int(s['images'])
 assert [int(i['sequence']) for i in images]==list(range(len(images)))
 assert int(s['max_inflight'])<=int(s['capacity'])
 for i in images:
  score=float(i['score']);assert np.isfinite(score) and abs(score-float(i['expected_score']))<=1e-6
  assert int(i['decision'])==int(score>float(i['threshold']))
  prep=float(i['read_ms'])+float(i['decode_preprocess_ms']);wait=float(i['queue_reorder_ms']);service=float(i['service_host_ms']);total=float(i['png_to_decision_ms'])
  assert min(prep,wait,service)>=0 and abs(prep+wait+service-total)<1e-6
 energy,_=integrate(read(directory/'power.csv'),int(s['start_ns']),int(s['end_ns']))
 recalculated=sum(energy.values())*1000/len(images)
 max_energy_error=max(max_energy_error,abs(recalculated-float(s['three_rail_mj_per_image'])))
 error_count+=len(json.loads((directory/'sensor_read_errors.json').read_text()));total_images+=len(images)
groups=collections.defaultdict(list)
for s in runs:groups[(s['category'],s['engine'],int(s['producers']),int(s['capacity']))].append(s)
fields=['images_per_second','png_to_decision_ms_p50','png_to_decision_ms_p95','three_rail_mj_per_image','three_rail_mean_w','gpu_event_ms_p50','prep_mean_ms','wait_mean_ms','service_mean_ms','prep_tail5_mean_ms','wait_tail5_mean_ms','service_tail5_mean_ms','total_tail5_mean_ms']
summary=[]
for (c,e,w,b),rs in sorted(groups.items()):
 s={'category':c,'engine':e,'producers':w,'capacity':b,'rounds':len(rs)}
 for field in fields:
  values=[float(r[field]) for r in rs]
  for stat,fn in [('mean',np.mean),('min',np.min),('max',np.max)]:s[field+'_'+stat]=float(fn(values))
 summary.append(s)
write('condition_summary.csv',summary)
contrasts=[]
for c in cats:
 for w,b in expected_cells:
  left=sorted(groups[(c,'default_tf32',w,b)],key=lambda x:int(x['round']));right=sorted(groups[(c,'automatic_fp16',w,b)],key=lambda x:int(x['round']))
  s={'category':c,'producers':w,'capacity':b,'pairs':rounds}
  for field,label,direction in [('images_per_second','throughput_gain',1),('three_rail_mj_per_image','energy_saving',-1)]:
   values=[direction*100*(float(y[field])/float(x[field])-1) for x,y in zip(left,right)]
   for stat,fn in [('mean',np.mean),('min',np.min),('max',np.max)]:s[label+'_pct_'+stat]=float(fn(values))
  contrasts.append(s)
write('paired_contrasts.csv',contrasts)
if mode=='full':
 predictions=[]
 for c in cats:
  for e in ('default_tf32','automatic_fp16'):
   base=next(r for r in summary if (r['category'],r['engine'],r['producers'],r['capacity'])==(c,e,1,10))
   for w in (2,4,6,8):
    actual=next(r for r in summary if (r['category'],r['engine'],r['producers'],r['capacity'])==(c,e,w,10))
    predicted=min(w*1000/base['prep_mean_ms_mean'],1000/base['service_mean_ms_mean']);observed=actual['images_per_second_mean']
    predictions.append({'category':c,'engine':e,'producers':w,'capacity':10,'predicted_img_s_from_P1':predicted,'observed_img_s':observed,'signed_error_pct':100*(predicted/observed-1)})
 write('bottleneck_predictions.csv',predictions)
audit={'status':'PASS','mode':mode,'runs_verified':len(runs),'timed_images':total_images,'score_decision_replay':'PASS','count_and_capacity':'PASS','max_energy_reintegration_error_mj_per_image':max_energy_error,'sensor_read_errors':error_count,'statistics_unit':'run; observed ranges are not confidence intervals'}
(root/'acquisition_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)
for s in contrasts:print(f"{s['category']}/P{s['producers']}/C{s['capacity']} throughput={s['throughput_gain_pct_mean']:+.3f}% energy={s['energy_saving_pct_mean']:+.3f}%",flush=True)
