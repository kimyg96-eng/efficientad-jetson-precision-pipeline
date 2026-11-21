import errno
import json
import re
import threading
import time
from pathlib import Path
import numpy as np
RAILS=('VDD_GPU_SOC','VDD_CPU_CV','VIN_SYS_5V0')

def integrate(samples,start_ns,end_ns):
    t=np.array([(int(s['monotonic_ns'])-start_ns)/1e9 for s in samples]);duration=(end_ns-start_ns)/1e9
    if not len(t)>2 or not t[0]<0<duration<t[-1] or not np.all(np.diff(t)>0):
        raise ValueError('Power samples do not strictly bracket workload')
    bounds=np.r_[0.,t[(t>0)&(t<duration)],duration]
    gap=float(np.diff(t)[(t[:-1]<duration)&(t[1:]>0)].max()*1000)
    if gap>150:raise ValueError(f'Power integration gap too large: {gap}ms')
    energies={}
    for rail in RAILS:
        watts=np.array([int(s[rail+'_mV'])*int(s[rail+'_mA'])/1e6 for s in samples])
        if not np.isfinite(watts).all() or np.any(watts<0):raise ValueError('Invalid power')
        energies[rail]=float(np.trapezoid(np.interp(bounds,t,watts),bounds))
    return energies,gap

class Monitor:
    def __init__(self,bounds):
        self.samples=[];self.telemetry=[];self.read_errors=[];self.errors=[]
        self.stop=threading.Event();self.ready=threading.Event();self.bounds=bounds
        self.sensors={}
        for label in sorted(Path('/sys/class/hwmon').glob('hwmon*/in*_label')):
            rail=label.read_text().strip();match=re.fullmatch(r'in(\d+)_label',label.name)
            if rail not in RAILS or match is None:continue
            if rail in self.sensors:raise ValueError('Duplicate power rail')
            self.sensors[rail]=(label.with_name(f'in{match[1]}_input'),label.with_name(f'curr{match[1]}_input'))
        if set(self.sensors)!=set(RAILS):raise ValueError('Missing power rails')
        self.paths={'gpu_hz':Path('/sys/class/devfreq/17000000.gpu/cur_freq'),
                    'emc_hz':Path('/sys/class/devfreq/bwmgr/cur_freq')}
        for t in bounds['targets']:
            p=Path(t['minimum'])
            if p.name=='scaling_min_freq':self.paths[p.parent.name+'_cpu_khz']=p.with_name('scaling_cur_freq')
        for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
            if (zone/'type').is_file() and any(k in (zone/'type').read_text().lower() for k in ('cpu','gpu','tj')):
                self.paths[zone.name+'_mC']=zone/'temp'
        for p in sorted(Path('/sys/class/hwmon').glob('hwmon*/pwm1')):self.paths[p.parent.name+'_pwm1']=p
        for pattern in ('hwmon*/rpm','hwmon*/fan*_input'):
            for p in sorted(Path('/sys/class/hwmon').glob(pattern)):self.paths[p.parent.name+'_'+p.name]=p
    def read(self,path,required):
        for attempt in range(3):
            try:return int(path.read_text())
            except OSError as exc:
                self.read_errors.append({'monotonic_ns':time.monotonic_ns(),'path':str(path),'errno':exc.errno,'attempt':attempt+1})
                if attempt<2 and exc.errno in (errno.EBUSY,errno.EAGAIN,errno.EINTR):time.sleep(.002);continue
                if required:raise
                return None
    def sample_power(self):
        before=time.monotonic_ns();row={}
        for rail in RAILS:
            v,c=self.sensors[rail];row[rail+'_mV']=self.read(v,True);row[rail+'_mA']=self.read(c,True)
            if row[rail+'_mV']<=0 or row[rail+'_mA']<0:raise ValueError('Invalid rail voltage/current')
        row={'monotonic_ns':(before+time.monotonic_ns())//2,**row}
        self.samples.append(row)
    def collect(self):
        deadline=time.monotonic_ns();last_extra=0
        try:
            while not self.stop.is_set():
                self.sample_power()
                now=time.monotonic_ns()
                if now-last_extra>=1_000_000_000:
                    row={'monotonic_ns':now,**{k:self.read(p,k in ('gpu_hz','emc_hz') or k.endswith('_cpu_khz')) for k,p in self.paths.items()}}
                    for t in self.bounds['targets']:
                        if self.read(Path(t['minimum']),True)!=t['original_max'] or self.read(Path(t['maximum']),True)!=t['original_max']:
                            raise RuntimeError('Frequency bounds changed during benchmark')
                    self.telemetry.append(row);last_extra=now
                self.ready.set()
                deadline+=50_000_000
                self.stop.wait(max(0,deadline-time.monotonic_ns())/1e9)
        except BaseException as exc:self.errors.append(exc);self.ready.set()
    def start(self):
        self.worker=threading.Thread(target=self.collect,name='power-observer',daemon=True);self.worker.start()
        if not self.ready.wait(5):raise RuntimeError('Power observer start timeout')
        self.check()
    def check(self):
        if self.errors:raise RuntimeError('Power/clock observer failed') from self.errors[0]
    def finish(self):
        self.stop.set();self.worker.join(timeout=5)
        if self.worker.is_alive():raise RuntimeError('Power observer did not stop')
        self.check();self.sample_power()
