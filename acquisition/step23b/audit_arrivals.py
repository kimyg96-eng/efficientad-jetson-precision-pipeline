import csv,json,sys,collections
from pathlib import Path
import numpy as np
from power_monitor import integrate
root=Path(sys.argv[1]);p=json.loads((root/'protocol.json').read_text());cats=['bottle','carpet'] if p['mode']=='pilot' else ['bottle','carpet','capsule','grid'];expected={(c,e,rate,r) for c in cats for e in ['default_tf32','automatic_fp16'] for rate in p['rates_img_s'] for r in range(1,p['rounds']+1)}
actual=set();curves=[];total=0;maxerr=0.;gaps=[];groups=collections.defaultdict(list)
for f in root.glob('*/*/rate_*/round_*/summary.json'):
 s=json.loads(f.read_text());key=s['category'],s['engine'],s['rate_img_s'],s['round'];assert key not in actual;actual.add(key)
 read=lambda n:list(csv.DictReader((f.parent/n).open()))
 images,arrivals=read('images.csv'),read('arrivals.csv');assert len(images)==s['completed'] and len(arrivals)==s['offered'];assert len(images)+sum(x['status']!='accepted' for x in arrivals)==len(arrivals)
 assert [int(x['sequence']) for x in images]==[int(x['sequence']) for x in arrivals if x['status']=='accepted']
 for x in images:
  assert abs(float(x['score'])-float(x['expected_score']))<=1e-6
  assert int(x['decision'])==int(float(x['score'])>float(x['threshold']))
 values=np.array([float(x['arrival_to_decision_ms']) for x in images])
 for b in range(20,221,5):curves.append({'category':key[0],'engine':key[1],'rate_img_s':key[2],'round':key[3],'deadline_ms':b,'on_time_pct_of_offered':100*int((values<=b).sum())/s['offered']})
 energy,gap=integrate(read('power.csv'),s['start_ns'],s['end_ns']);maxerr=max(maxerr,abs(sum(energy.values())-s['three_rail_energy_j']));gaps.append(gap);total+=len(images);groups[key[:3]].append(s)
assert actual==expected
with (root/'deadline_curves.csv').open('w',newline='') as out:
 w=csv.DictWriter(out,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
rates=[]
for c in cats:
 for e in ['default_tf32','automatic_fp16']:
  good=[r for r in p['rates_img_s'] if all(s['drop_capacity']==0 and s['drop_release_late']==0 for s in groups[(c,e,r)])]
  rates.append({'category':c,'engine':e,'largest_tested_zero_drop_rate':max(good) if good else None,'tested_rates':p['rates_img_s'],'scope':'finite observations at this grid; no extrapolation; completed drops include scheduler-late releases'})
(root/'zero_drop_summary.json').write_text(json.dumps(rates,indent=2)+'\n')
print(json.dumps({'status':'PASS','runs':len(actual),'completed':total,'max_energy_error_j':maxerr,'max_power_gap_ms':max(gaps),'mode':p['mode']}),flush=True)
