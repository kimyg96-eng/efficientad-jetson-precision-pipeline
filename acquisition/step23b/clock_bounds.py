"""Clamp only selected min-frequency bounds and restore verified fields."""
import json
import os
import sys
import time
from pathlib import Path

def wait_bound(minimum, maximum, requested, expected_max, timeout=3.0):
    """Require two matching readbacks; sysfs updates may be asynchronous."""
    deadline=time.monotonic()+timeout
    readings=[]; consecutive=0
    while True:
        actual=int(Path(minimum).read_text())
        actual_max=int(Path(maximum).read_text())
        readings.append(actual)
        if actual_max!=expected_max:
            raise RuntimeError(f'Maximum changed: {maximum}: {actual_max} != {expected_max}')
        consecutive=consecutive+1 if actual==requested else 0
        if consecutive>=2:
            print(json.dumps({'path':minimum,'requested':requested,'readbacks':readings,'status':'PASS'}),flush=True)
            return actual
        if time.monotonic()>=deadline:
            raise RuntimeError(f'Frequency-bound timeout: path={minimum} requested={requested} readbacks={readings}')
        time.sleep(0.1)

def discover(root=Path('/')):
    def path(relative): return root / relative
    online_text=path('sys/devices/system/cpu/online').read_text().strip()
    online=set()
    for part in online_text.split(','):
        if '-' in part:
            a,b=map(int,part.split('-'));online.update(range(a,b+1))
        else:online.add(int(part))
    targets=[]
    for policy in sorted(path('sys/devices/system/cpu/cpufreq').glob('policy*')):
        if int(policy.name[6:]) in online:
            targets.append({'minimum':str(policy/'scaling_min_freq'),
                            'maximum':str(policy/'scaling_max_freq'), 'expected_max':1728000})
    if not targets:raise ValueError('No active CPU policies')
    targets.extend([
        {'minimum':str(path('sys/class/devfreq/17000000.gpu/min_freq')),
         'maximum':str(path('sys/class/devfreq/17000000.gpu/max_freq')), 'expected_max':612000000},
        {'minimum':str(path('sys/class/devfreq/bwmgr/min_freq')),
         'maximum':str(path('sys/class/devfreq/bwmgr/max_freq')), 'expected_max':3199000000}])
    for t in targets:
        t['original_min']=int(Path(t['minimum']).read_text())
        t['original_max']=int(Path(t['maximum']).read_text())
        if t['original_max']!=t['expected_max'] or not 0<=t['original_min']<=t['original_max']:
            raise ValueError(f'Unexpected MODE_30W bounds: {t}')
    return targets,online_text

def restore(state, allowed):
    allow={(x['minimum'],x['maximum']) for x in allowed}
    checks=[]
    for t in reversed(state['targets']):
        if (t['minimum'],t['maximum']) not in allow:
            raise ValueError('Restore path outside frequency-bound allowlist')
        current_max=int(Path(t['maximum']).read_text())
        if current_max!=t['original_max']:
            raise ValueError('Maximum changed externally; not overwriting other settings')
        current_min=int(Path(t['minimum']).read_text())
        if current_min not in (t['original_min'],t['original_max']):
            raise ValueError('Minimum changed externally; restoration needs review')
        Path(t['minimum']).write_text(str(t['original_min']))
        actual=wait_bound(t['minimum'],t['maximum'],t['original_min'],t['original_max'])
        checks.append({'path':t['minimum'],'restored':actual,'max_unchanged':current_max})
    return {'status':'PASS_SELECTED_BOUNDS_READ_BACK','checks':checks}

def apply(targets, state_path, online):
    state={'scope':'selected CPU/GPU/EMC minimum bounds only; maxima, governors, idle states, fan and power mode untouched',
           'online_cpus':online,'targets':targets}
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    os.chmod(state_path,0o644)
    try:
        for t in targets:
            Path(t['minimum']).write_text(str(t['original_max']))
            wait_bound(t['minimum'],t['maximum'],t['original_max'],t['original_max'])
    except BaseException:
        restore(state,targets)
        raise
    return state

if __name__=='__main__':
    command,out=sys.argv[1],Path(sys.argv[2]);state_path=out/'original_frequency_bounds.json'
    targets,online=discover()
    if command=='apply':
        if state_path.exists():raise ValueError('Refusing to replace an existing bounds snapshot')
        apply(targets,state_path,online)
        print('CLOCK_MIN_BOUNDS_APPLIED_AND_READ_BACK=PASS')
    elif command=='restore':
        state=json.loads(state_path.read_text())
        if online!=state['online_cpus']:raise ValueError('Online CPU set changed')
        report=restore(state,targets)
        (out/'frequency_bounds_restore.json').write_text(json.dumps(report,indent=2)+'\n')
        os.chmod(out/'frequency_bounds_restore.json',0o644)
        print('CLOCK_MIN_BOUNDS_RESTORE=PASS_SELECTED_FIELDS_READ_BACK')
    else:raise ValueError('Unknown clock-bounds action')
