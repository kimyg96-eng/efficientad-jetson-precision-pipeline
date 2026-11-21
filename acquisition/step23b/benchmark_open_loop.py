"""One TensorRT consumer; ordered 0/1/2/4 CPU producers; averaged rail energy."""
import argparse,ast,csv,hashlib,io,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pipeline_core import run_ordered
from power_monitor import Monitor,integrate,RAILS

parser=argparse.ArgumentParser()
parser.add_argument('audit',type=Path);parser.add_argument('out',type=Path);parser.add_argument('trt_site',type=Path)
parser.add_argument('--mode',choices=['preflight','pilot','full'],default='pilot')
args=parser.parse_args();AUDIT=args.audit.resolve();OUT=args.out.resolve();PKG=Path(__file__).resolve().parent
import sys
sys.path.append(str(args.trt_site))
import tensorrt as trt
if trt.__version__!='10.16.2.10':raise RuntimeError('TensorRT version changed')
torch.set_num_threads(1);torch.set_num_interop_threads(1)
DATA=AUDIT.parent/'04_Datasets/MVTec_AD/01_Extracted'

def readcsv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def writecsv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def mode_check():
    result=subprocess.check_output(['/usr/sbin/nvpmodel','-q'],text=True)
    if 'NV Power Mode: MODE_30W' not in result:raise ValueError('Expected MODE_30W')
    return result

frozen=readcsv(PKG/'frozen_inputs.csv');replay=readcsv(PKG/'replay.csv');engines=json.loads((PKG/'engines.json').read_text())
if len(frozen)!=410 or len(replay)!=410:raise ValueError('Unexpected frozen inventory size')
by_path={r['path']:r for r in frozen}
if len(by_path)!=410 or {r['path'] for r in replay}!=set(by_path):raise ValueError('Frozen/replay inventory mismatch')
for row in frozen:
    path=(DATA/row['path']).resolve(strict=True)
    if not path.is_relative_to((DATA/row['category']/'test').resolve()):raise ValueError('Unexpected image path')
    if sha(path)!=row['sha256']:raise ValueError(f'Input hash mismatch: {path}')
for category,vs in engines.items():
    for name,item in vs.items():
        path=(AUDIT/item['relative_path']).resolve(strict=True)
        if not path.is_relative_to(AUDIT):raise ValueError('Engine outside audit root')
        if sha(path)!=item['sha256']:raise ValueError(f'Engine hash mismatch: {path}')
        item['resolved_path']=str(path)
counts={c:sum(r['category']==c for r in replay) for c in engines}
if counts!={'bottle':83,'carpet':117,'capsule':132,'grid':78}:raise ValueError('Replay category mismatch')
for c in engines:
    for name in ('default_tf32','automatic_fp16'):
        if len({r[name+'_threshold'] for r in replay if r['category']==c})!=1:raise ValueError('Threshold changes within category')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'preflight.json').write_text(json.dumps({'images_sha256_verified':410,'engines_sha256_verified':8,'mode':mode_check(),'torch':torch.__version__,'tensorrt':trt.__version__,'torch_intra_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'engines':engines},indent=2)+'\n')
print('PREFLIGHT=PASS / IMAGES=410 / ENGINES=8 / TORCH_THREADS=1',flush=True)
if args.mode=='preflight':sys.exit(0)


from open_loop_core import run_arrivals
bounds=json.loads((OUT/'original_frequency_bounds.json').read_text())
for t in bounds['targets']:
    if int(Path(t['minimum']).read_text())!=t['original_max'] or int(Path(t['maximum']).read_text())!=t['original_max']:raise ValueError('Clock bounds not clamped')
CATEGORIES=('bottle','carpet') if args.mode=='pilot' else ('bottle','carpet','capsule','grid')
RATES=(35,80) if args.mode=='pilot' else (35,45,55,70,80)
ROUNDS=1 if args.mode=='pilot' else 6
DURATION=10 if args.mode=='pilot' else 60
WARMUP=2 if args.mode=='pilot' else 10
SETTLE=2 if args.mode=='pilot' else 5
input_shape=(1,3,256,256);output_names=('pred_score','anomaly_map')
runtime=trt.Runtime(trt.Logger(trt.Logger.WARNING))
exec(compile(ast.parse((PKG/'engine_function.py').read_text()),str(PKG/'engine_function.py'),'exec'))
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
protocol={'scope':'synthetic fixed-rate PNG arrivals; no camera capture','mode':args.mode,'rates_img_s':RATES,'rounds':ROUNDS,'offered_seconds':DURATION,'producers':4,'capacity':5,'capacity_scope':'accepted pending + preprocessing + ordered ready + consumer','drop_policy':'drop new when capacity full; drop scheduled release if dispatch lateness >= one period; no burst catchup','deadline_primary':'none: no application-derived deadline; report full exploratory 20-220ms curve','deadline_sensitivity_ms':[50,75,100,125,150,175,200], 'period_fields':'retained as legacy diagnostics only, not application deadlines', 'grid_extension':'five new rates; do not pool runs with older 15/30/60 sessions', 'rate_order':'rotate one rate position per round; reverse engine order each round; six rounds do not perfectly balance five rate positions', 'zero_drop_definition':'all offered arrivals completed in every observed round at a tested rate; no extrapolated continuous capacity claim','latency':'scheduled arrival to scalar decision, including dispatch lag and admission queue','arrival_order':'single ordered consumer preserves admitted sequence order','energy':'three rails from scheduled first arrival through drain/join; no idle subtraction; report per offered, completed and primary on-time image','rate_meaning':'synthetic workloads, not requirements inferred from results','warmup_seconds':WARMUP,'settle_seconds':SETTLE,'clock_policy':bounds,'statistical_unit':'run, not repeated images; no bootstrap in acquisition','no_stop_early':'all prespecified full conditions run regardless of observed results'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
def preprocess(row,seq):
    start=time.monotonic_ns();payload=(DATA/row['path']).read_bytes();read_end=time.monotonic_ns()
    with torch.inference_mode():
        with Image.open(io.BytesIO(payload)) as image:tensor=transform(image.convert('RGB')).unsqueeze(0).contiguous()
    return row,tensor,seq,start,read_end,time.monotonic_ns()
all_stats=[]
with torch.inference_mode():
 for category in CATEGORIES:
    data_rows=[r for r in replay if r['category']==category]
    runners={n:open_engine(Path(e['resolved_path']),category) for n,e in engines[category].items()}
    stream=torch.cuda.Stream();ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True);ev2=torch.cuda.Event(enable_timing=True)
    def consume(item,name,monitor=None):
        if monitor is not None:monitor.check()
        row,image,seq,start,read_end,ready=item;service=time.monotonic_ns();_,ctx,buf=runners[name]
        with torch.cuda.stream(stream):
            ev0.record(stream);buf['image'].copy_(image);ev1.record(stream)
            if not ctx.execute_async_v3(stream.cuda_stream):raise RuntimeError('TensorRT failed')
            ev2.record(stream)
        stream.synchronize();score=float(buf['pred_score'].cpu().item());threshold=float(row[name+'_threshold']);decision=int(score>threshold);finished=time.monotonic_ns()
        if not np.isfinite(score) or abs(score-float(row[name]))>1e-6 or decision!=int(float(row[name])>threshold):raise ValueError('Score/decision replay failed')
        return {'read_start_ns':start,'read_end_ns':read_end,'preprocess_end_ns':ready,'service_start_ns':service,'finished_ns':finished,'score':score,'expected_score':float(row[name]),'threshold':threshold,'decision':decision,'read_ms':(read_end-start)/1e6,'preprocess_ms':(ready-read_end)/1e6,'ready_wait_ms':(service-ready)/1e6,'service_ms':(finished-service)/1e6,'h2d_event_ms':ev0.elapsed_time(ev1),'gpu_event_ms':ev1.elapsed_time(ev2)}
    for round_id in range(1,ROUNDS+1):
        rates=list(RATES);offset=(round_id-1)%len(rates);rates=rates[offset:]+rates[:offset]
        engine_order=('default_tf32','automatic_fp16') if round_id%2 else ('automatic_fp16','default_tf32')
        for rate in rates:
          for name in engine_order:
            directory=OUT/category/name/f'rate_{rate}'/f'round_{round_id}';directory.mkdir(parents=True)
            mode_check();time.sleep(SETTLE)
            run_ordered(data_rows,4,5,WARMUP,preprocess,lambda item:consume(item,name))
            monitor=Monitor(bounds);partial=[];outputs=[];arrivals=[]
            def measured(item):
                result=consume(item,name,monitor);partial.append(result);return result
            try:
                monitor.start();outputs,arrivals,times=run_arrivals(data_rows,rate,DURATION,4,5,preprocess,measured);monitor.finish()
                energy,gap=integrate(monitor.samples,times['start_ns'],times['end_ns']);joules=sum(energy.values())
                offered=times['offered'];completed=len(outputs);dropped=sum(x['status']!='accepted' for x in arrivals)
                assert completed+dropped==offered
                ontime=sum(x['on_time_period'] for x in outputs);ontime100=sum(x['on_time_100ms'] for x in outputs)
                record={'category':category,'engine':name,'rate_img_s':rate,'round':round_id,'offered':offered,'completed':completed,'drop_capacity':sum(x['status']=='drop_capacity' for x in arrivals),'drop_release_late':sum(x['status']=='drop_release_late' for x in arrivals),'drop_pct':100*dropped/offered,'period_deadline_ms':1000/rate,'on_time_period':ontime,'on_time_period_pct_of_offered':100*ontime/offered,'deadline_miss_pct_of_completed':100*(completed-ontime)/completed if completed else None,'on_time_100ms':ontime100,'on_time_100ms_pct_of_offered':100*ontime100/offered,'delivered_during_arrival_window_img_s':sum(x['finished_ns']<=times['arrival_end_ns'] for x in outputs)/DURATION,'completed_with_drain_img_s':completed/((times['end_ns']-times['start_ns'])/1e9),'start_ns':times['start_ns'],'arrival_end_ns':times['arrival_end_ns'],'end_ns':times['end_ns'],'max_inflight':times['max_inflight'],'three_rail_energy_j':joules,'mean_three_rail_w':joules/((times['end_ns']-times['start_ns'])/1e9),'mj_per_offered':joules*1000/offered,'mj_per_completed':joules*1000/completed if completed else None,'mj_per_period_on_time':joules*1000/ontime if ontime else None,'max_power_gap_ms':gap,'replay':'PASS'}
                for field in ('arrival_to_decision_ms','release_lateness_ms','ready_wait_ms','service_ms','gpu_event_ms'):
                    vals=[x[field] for x in outputs]
                    for q in (50,95,99):record[field+f'_p{q}']=float(np.percentile(vals,q)) if vals else None
                for budget in (50,75,100,125,150,175,200):
                    count=sum(x['arrival_to_decision_ms']<=budget for x in outputs);record[f'on_time_{budget}ms']=count;record[f'on_time_{budget}ms_pct_of_offered']=100*count/offered
                (directory/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');all_stats.append(record);writecsv(OUT/'per_run.csv',all_stats)
            finally:
                if hasattr(monitor,'worker') and not monitor.stop.is_set():
                    try:monitor.finish()
                    except BaseException:pass
                writecsv(directory/'images.csv',partial);writecsv(directory/'arrivals.csv',arrivals);writecsv(directory/'power.csv',monitor.samples);writecsv(directory/'telemetry.csv',monitor.telemetry)
                (directory/'sensor_read_errors.json').write_text(json.dumps(monitor.read_errors,indent=2)+'\n')
            print(f"RUN_COMPLETE={category}/{name}/RATE{rate}/R{round_id} OFFERED={offered} DONE={completed} DROP_PCT={record['drop_pct']:.3f} DELIVERY_100MS_PCT={record['on_time_100ms_pct_of_offered']:.3f}",flush=True)
    del consume,runners
for row in frozen:
    if sha(DATA/row['path'])!=row['sha256']:raise ValueError('Input changed')
expected_runs=len(CATEGORIES)*len(RATES)*2*ROUNDS
if len(all_stats)!=expected_runs:raise ValueError('Incomplete run matrix')
(OUT/'summary.json').write_text(json.dumps({'status':'COMPLETE','mode':args.mode,'runs':len(all_stats),'expected_runs':expected_runs,'replay':'PASS','pipeline_runs':all_stats},indent=2,allow_nan=False)+'\n')
print(f'STEP23B COMPLETE / MODE={args.mode} / RUNS={len(all_stats)} / REPLAY=PASS',flush=True)
