import hashlib,json,sys
from pathlib import Path

audit,out=map(Path,sys.argv[1:]);build=audit/'STEP16A_SELECTIVE_FP16_TF32_BUILD_ZZe4JJ4H';refs=audit/'STEP13A_R1_EFFICIENTAD_TF32_REFERENCES_G2Jd0c1J'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_manifest(root):
 entries={}
 for line in (root/'SHA256SUMS.txt').read_text().splitlines():
  if not line.strip():continue
  digest,rel=line.split(maxsplit=1);rel=rel.lstrip('*').removeprefix('./');entries[rel]=digest
 return entries
job_manifest=check_manifest(build);ref_manifest=check_manifest(refs)
summary=build/'summary.json'
if job_manifest.get('summary.json')!=sha(summary):raise ValueError('Build summary hash mismatch')
jobs=json.loads(summary.read_text())['completed']
expected={'all_conv_fp16','teacher_conv_fp16','student_conv_fp16','autoencoder_conv_fp16','ae_encoder_conv_fp16','ae_decoder_conv_fp16'}
if {(j['category'],j['configuration']) for j in jobs}!={(c,n) for c in ('bottle','carpet') for n in expected} or len(jobs)!=12:raise ValueError('Unexpected engine inventory')
engines=[]
refdir=out/'strong_reference';refdir.mkdir()
for c in ('bottle','carpet'):
 rel=f'{c}/strongly_typed_tf32_allowed/model.engine';path=refs/rel
 if ref_manifest.get(rel)!=sha(path):raise ValueError(f'Reference engine hash mismatch: {path}')
 command=refs/c/'strongly_typed_tf32_allowed/build_command.txt'
 cmd=command.read_text()
 if '--stronglyTyped' not in cmd or '--noTF32' in cmd:raise ValueError('Reference build policy mismatch')
 if ref_manifest.get(command.relative_to(refs).as_posix())!=sha(command):raise ValueError('Reference build command hash mismatch')
 (refdir/f'{c}_efficientad_s_fp32.engine').symlink_to(path.resolve(strict=True))
 engines.append(dict(category=c,configuration='strong_tf32',path=str(path),sha256=sha(path)))
for job in jobs:
 path=build/job['engine']
 if sha(path)!=job['engine_sha256']:raise ValueError(f'Candidate hash mismatch: {path}')
 engines.append(dict(category=job['category'],configuration=job['configuration'],path=str(path),sha256=sha(path)))
# Check frozen helper scripts used for the engine interface and image loading.
for folder,name in [('STEP11J_EFFICIENTAD_TRT_FP16_VAL_kygZcMS9','verify_fp16.py'),('STEP11N_EFFICIENTAD_AE_SPLIT_FP16_B1HhoFFr','verify_conv_fp16.py')]:
 root=audit/folder;manifest=check_manifest(root)
 if manifest.get(name)!=sha(root/name):raise ValueError(f'Helper hash mismatch {root/name}')
shared=audit/'STEP13E_EFFICIENTAD_MATCHED_GPU_LATENCY_KtNxqU58/shared_input.bin'
if shared.stat().st_size!=3*256*256*4:raise ValueError('Shared input shape mismatch')
(out/'engine_inventory.json').write_text(json.dumps({'reference_policy':'strongly typed FP32 tensors with TF32 allowed; identical typing mode to explicit candidates','engines':engines,'shared_input_sha256':sha(shared)},indent=2)+'\n')
(out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
print('ENGINE_INTEGRITY=14/14 / REFERENCES_STRONGLY_TYPED_TF32=2/2 / NO_BUILD_REQUIRED')
