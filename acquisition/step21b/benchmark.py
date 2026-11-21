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
import PIL, torchvision
inventory=[]
for row in frozen:
    with Image.open(DATA/row['path']) as image:
        inventory.append({'path':row['path'],'width':image.width,'height':image.height,'format':image.format,'mode':image.mode})
writecsv(OUT/'image_inventory.csv',inventory)
(OUT/'implementation.json').write_text(json.dumps({'pillow':PIL.__version__,'torchvision':torchvision.__version__,'torch':torch.__version__,'online_cpus':Path('/sys/devices/system/cpu/online').read_text().strip(),'process_affinity':sorted(os.sched_getaffinity(0)),'affinity_modified_by_package':False,'gil_release':'not inferred from throughput; not measured directly','decode':'Pillow Image.open then RGB conversion; torchvision Resize and ToTensor'},indent=2)+'\n')
if args.mode=='preflight':sys.exit(0)

bounds=json.loads((OUT/'original_frequency_bounds.json').read_text())
for t in bounds['targets']:
    if int(Path(t['minimum']).read_text())!=t['original_max'] or int(Path(t['maximum']).read_text())!=t['original_max']:
        raise ValueError('Clock minimum bounds are not clamped as recorded')
ROUNDS=1 if args.mode=='pilot' else 8
DURATION=10 if args.mode=='pilot' else 60
WARMUP=3 if args.mode=='pilot' else 10
SETTLE=2 if args.mode=='pilot' else 5
CATEGORIES=('bottle','carpet') if args.mode=='pilot' else ('bottle','carpet','capsule','grid')
protocol={'mode':args.mode,'rounds':ROUNDS,'categories':CATEGORIES,'producers':[0,1,2,4],
          'inflight_capacity':5,'inflight_scope':'preprocessing + ready/reorder buffer + current inference; acquired before PNG read and released after decision',
          'order':'ordered consumer; producer completion may be out of order; no drop on successful run',
          'minimum_duration_s':DURATION,'warmup_seconds':WARMUP,'settle_seconds':SETTLE,
          'duration_policy':'complete repeated corpus passes; no new pass is assigned once the minimum duration has elapsed; assigned pass drains completely',
          'energy_window':'host workload start before worker startup through ordered completion and worker join; includes fill/drain and CPU preprocessing; average three-rail energy per completed image',
          'latency_scope':'PNG read start to scalar decision including preprocessor completion reorder waiting; excludes camera/external arrivals',
          'gpu_scope':'CUDA events around execute_async_v3 on the same stream after H2D',
          'cache_policy':'normal filesystem cache; hashes and warmup read images before measurement',
          'instrumentation':'CUDA event queries, record construction and replay checks occur after the decision timestamp but within the overall workload/energy window; power polling is active',
          'torch_threads':{'intra':1,'interop':1},'clock_policy':bounds['scope'],
          'clock_targets':bounds['targets'],'idle_and_fan_policy':'not changed by this package; record exposed fan/thermal sensors',
          'pilot_scope':'execution validation only; not a replacement for full eight-round measurements',
          'arm_order':'producer blocks rotate every two rounds; engine order reverses within each block every round; each engine occupies every within-round position once over eight rounds',
          'provenance':json.loads((PKG/'provenance.json').read_text())}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
input_shape=(1,3,256,256);output_names=('pred_score','anomaly_map')
runtime=trt.Runtime(trt.Logger(trt.Logger.WARNING))
exec(compile(ast.parse((PKG/'engine_function.py').read_text()),str(PKG/'engine_function.py'),'exec'))
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
all_stats=[]

def preprocess(row,sequence):
    started=time.monotonic_ns();payload=(DATA/row['path']).read_bytes();read_end=time.monotonic_ns()
    with torch.inference_mode():
        with Image.open(io.BytesIO(payload)) as image:
            tensor=transform(image.convert('RGB')).unsqueeze(0).contiguous()
    ready=time.monotonic_ns()
    return row,tensor,sequence,started,read_end,ready

with torch.inference_mode():
    for category in CATEGORIES:
        rows=[r for r in replay if r['category']==category]
        runners={name:open_engine(Path(item['resolved_path']),category) for name,item in engines[category].items()}
        stream=torch.cuda.Stream();begin=torch.cuda.Event(enable_timing=True);gpu_start=torch.cuda.Event(enable_timing=True);gpu_end=torch.cuda.Event(enable_timing=True)
        def consume(item,name,monitor):
            if monitor is not None:monitor.check()
            row,image,sequence,started,read_end,ready=item
            service_start=time.monotonic_ns();_,ctx,buf=runners[name]
            with torch.cuda.stream(stream):
                begin.record(stream);buf['image'].copy_(image);gpu_start.record(stream)
                if not ctx.execute_async_v3(stream.cuda_stream):raise RuntimeError('TensorRT execution failed')
                gpu_end.record(stream)
            stream.synchronize();sync_end=time.monotonic_ns()
            score=float(buf['pred_score'].cpu().item());threshold=float(row[name+'_threshold']);decision=int(score>threshold);finished=time.monotonic_ns()
            expected=float(row[name])
            if not np.isfinite(score) or abs(score-expected)>1e-6 or decision!=int(expected>threshold):
                raise ValueError(f'Score/decision replay failed: {category}/{name}/{row["path"]}: {score} vs {expected}')
            return {'sequence':sequence,'path':row['path'],'score':score,'expected_score':expected,'score_delta':score-expected,'threshold':threshold,'decision':decision,
                    'read_start_ns':started,'read_end_ns':read_end,'preprocess_end_ns':ready,'service_start_ns':service_start,'finished_ns':finished,
                    'read_ms':(read_end-started)/1e6,'decode_preprocess_ms':(ready-read_end)/1e6,'queue_reorder_ms':(service_start-ready)/1e6,
                    'h2d_event_ms':begin.elapsed_time(gpu_start),'gpu_event_ms':gpu_start.elapsed_time(gpu_end),
                    'score_decision_ms':(finished-sync_end)/1e6,'service_host_ms':(finished-service_start)/1e6,'png_to_decision_ms':(finished-started)/1e6}
        for round_id in range(1,ROUNDS+1):
            levels=[0,1,2,4];offset=((round_id-1)//2)%4;levels=levels[offset:]+levels[:offset]
            engine_order=('default_tf32','automatic_fp16') if round_id%2 else ('automatic_fp16','default_tf32')
            order=[(name,w) for w in levels for name in engine_order]
            for position,(name,workers) in enumerate(order,1):
                directory=OUT/category/name/f'producers_{workers}'/f'round_{round_id}';directory.mkdir(parents=True)
                mode_check();time.sleep(SETTLE)
                run_ordered(rows,workers,5,WARMUP,preprocess,lambda item:consume(item,name,None))
                monitor=Monitor(bounds);current=[];record=None
                def measured(item):
                    result=consume(item,name,monitor);current.append(result);return result
                try:
                    monitor.start()
                    outputs,times=run_ordered(rows,workers,5,DURATION,preprocess,measured)
                    monitor.finish()
                    if len(outputs)!=len(current) or [r['sequence'] for r in outputs]!=list(range(len(outputs))):raise ValueError('Incomplete/reordered workload')
                    if any(r['path']!=rows[i%len(rows)]['path'] for i,r in enumerate(outputs)):raise ValueError('Input order changed')
                    energy,gap=integrate(monitor.samples,times['start_ns'],times['end_ns'])
                    duration=(times['end_ns']-times['start_ns'])/1e9
                    if duration<DURATION:raise ValueError('Minimum measurement duration not reached')
                    record={'category':category,'engine':name,'producers':workers,'round':round_id,'order_position':position,'images':len(outputs),'corpus_passes':len(outputs)//len(rows),
                            'start_ns':times['start_ns'],'end_ns':times['end_ns'],'duration_s':duration,'images_per_second':len(outputs)/duration,
                            'max_inflight':times['max_inflight'],'three_rail_mj_per_image':sum(energy.values())*1000/len(outputs),'three_rail_mean_w':sum(energy.values())/duration,
                            'max_power_gap_ms':gap,'max_abs_score_delta':max(abs(r['score_delta']) for r in outputs),'replay':'PASS'}
                    for rail in RAILS:record[rail+'_energy_j']=energy[rail]
                    for field in ('png_to_decision_ms','read_ms','decode_preprocess_ms','queue_reorder_ms','h2d_event_ms','gpu_event_ms','score_decision_ms','service_host_ms'):
                        values=np.array([r[field] for r in outputs])
                        if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Invalid timing')
                        for q in (50,95,99):record[field+f'_p{q}']=float(np.percentile(values,q))
                    (directory/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
                    all_stats.append(record);writecsv(OUT/'per_run.csv',all_stats)
                finally:
                    if hasattr(monitor,'worker') and not monitor.stop.is_set():
                        try:monitor.finish()
                        except BaseException:pass
                    writecsv(directory/'images.csv',current);writecsv(directory/'power.csv',monitor.samples);writecsv(directory/'telemetry.csv',monitor.telemetry)
                    (directory/'sensor_read_errors.json').write_text(json.dumps(monitor.read_errors,indent=2)+'\n')
                print(f"RUN_COMPLETE={category}/{name}/P{workers}/R{round_id} IMG_S={record['images_per_second']:.3f} P95_MS={record['png_to_decision_ms_p95']:.3f} GPU_P50_MS={record['gpu_event_ms_p50']:.3f} ENERGY_MJ={record['three_rail_mj_per_image']:.3f}",flush=True)
        del consume,runners
for row in frozen:
    if sha(DATA/row['path'])!=row['sha256']:raise ValueError('Input changed during benchmark')
expected_runs=len(CATEGORIES)*8*ROUNDS
if len(all_stats)!=expected_runs:raise ValueError('Incomplete experiment')
(OUT/'summary.json').write_text(json.dumps({'status':'COMPLETE','mode':args.mode,'runs':len(all_stats),'expected_runs':expected_runs,'score_and_decision_replay':'PASS','energy_scope':'three onboard rails including PNG preprocessing, bounded queue and scalar-decision workload; no idle subtraction','pipeline_runs':all_stats},indent=2,allow_nan=False)+'\n')
print(f'STEP20A COMPLETE / MODE={args.mode} / RUNS={len(all_stats)} / REPLAY=PASS',flush=True)
