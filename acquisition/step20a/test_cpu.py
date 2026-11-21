"""Preparation fixtures only; no Jetson or TensorRT results are generated."""
import copy,json,tempfile,time,threading,unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core import run_ordered
from clock_bounds import discover,apply,restore,wait_bound
from power_monitor import integrate,RAILS

class PreparationTests(unittest.TestCase):
    def test_delayed_bound_readback(self):
        values=iter(['729600','1728000','1728000'])
        def delayed(path,*args,**kwargs):
            return next(values) if str(path)=='minimum' else '1728000'
        with patch.object(Path,'read_text',delayed),patch('clock_bounds.time.sleep'):
            self.assertEqual(wait_bound('minimum','maximum',1728000,1728000),1728000)
    def test_bound_timeout_names_path(self):
        with patch.object(Path,'read_text',lambda p: '1' if str(p)=='minimum' else '2'):
            with self.assertRaisesRegex(RuntimeError,'path=minimum requested=2'):
                wait_bound('minimum','maximum',2,2,timeout=0)
    def test_order_bound_complete_passes(self):
        for workers in (0,1,2,4):
            def prep(row,seq):
                time.sleep(.0025 if seq%3==0 else .0002)
                return seq,row
            def consume(item):time.sleep(.0003);return item
            results,meta=run_ordered(list(range(5)),workers,5,.025,prep,consume)
            self.assertGreaterEqual(meta['end_ns']-meta['start_ns'],25_000_000)
            self.assertEqual([x[0] for x in results],list(range(len(results))))
            self.assertEqual([x[1] for x in results],[i%5 for i in range(len(results))])
            self.assertEqual(len(results)%5,0)
            self.assertLessEqual(meta['max_inflight'],5)
            self.assertEqual(meta['assigned'],len(results))
    def test_producer_failure_joins(self):
        def prep(row,seq):
            if seq==3:raise ValueError('fixture producer failure')
            return seq
        with self.assertRaisesRegex(RuntimeError,'worker failed'):
            run_ordered([0,1],4,4,.1,prep,lambda x:x)
        self.assertFalse(any(t.name.startswith('png-producer') for t in threading.enumerate()))
    def test_consumer_failure_joins(self):
        def consume(seq):
            if seq==3:raise ValueError('fixture consumer failure')
            return seq
        with self.assertRaisesRegex(ValueError,'fixture consumer failure'):
            run_ordered([0,1],4,4,.1,lambda row,seq:seq,consume)
        self.assertFalse(any(t.name.startswith('png-producer') for t in threading.enumerate()))
    def test_power_integral(self):
        for linear in (False,True):
            samples=[]
            for tick in range(-1,22):
                t=tick*.05;row={'monotonic_ns':round(t*1e9)}
                for rail in RAILS:row[rail+'_mV']=1000;row[rail+'_mA']=round((1+t)*1000) if linear else 2000
                samples.append(row)
            energy,gap=integrate(samples,0,1_000_000_000)
            self.assertLessEqual(gap,50.000001)
            for v in energy.values():self.assertAlmostEqual(v,1.5 if linear else 2,places=10)
        with self.assertRaisesRegex(ValueError,'bracket'):
            integrate(samples,0,2_000_000_000)
    def fixture_bounds(self,root):
        f=root/'sys/devices/system/cpu/online';f.parent.mkdir(parents=True);f.write_text('0-7')
        for policy in (0,4):
            p=root/f'sys/devices/system/cpu/cpufreq/policy{policy}';p.mkdir(parents=True)
            (p/'scaling_min_freq').write_text('729600');(p/'scaling_max_freq').write_text('1728000')
        for name,lo,hi in [('17000000.gpu',306000000,612000000),('bwmgr',204000000,3199000000)]:
            p=root/f'sys/class/devfreq/{name}';p.mkdir(parents=True);(p/'min_freq').write_text(str(lo));(p/'max_freq').write_text(str(hi))
        return discover(root)
    def test_bounds_restore_and_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);targets,online=self.fixture_bounds(root)
            state=apply(targets,root/'saved.json',online)
            self.assertTrue(all(int(Path(t['minimum']).read_text())==t['original_max'] for t in targets))
            report=restore(state,targets);self.assertEqual(report['status'],'PASS_SELECTED_BOUNDS_READ_BACK')
            self.assertTrue(all(int(Path(t['minimum']).read_text())==t['original_min'] for t in targets))
            tampered=copy.deepcopy(state);tampered['targets'][0]['minimum']=str(root/'arbitrary')
            with self.assertRaisesRegex(ValueError,'allowlist'):restore(tampered,targets)
    def test_bounds_partial_apply_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);targets,online=self.fixture_bounds(root);original=Path.write_text;failpath=targets[2]['minimum'];failed=[]
            def fail_once(path,text,*args,**kwargs):
                if str(path)==failpath and not failed:
                    failed.append(True);raise OSError('fixture apply failure')
                return original(path,text,*args,**kwargs)
            with patch.object(Path,'write_text',fail_once):
                with self.assertRaisesRegex(OSError,'fixture apply failure'):apply(targets,root/'saved.json',online)
            self.assertTrue(all(int(Path(t['minimum']).read_text())==t['original_min'] for t in targets))
    def test_arm_order_balance(self):
        positions={}
        for r in range(1,9):
            levels=[0,1,2,4];offset=((r-1)//2)%4;levels=levels[offset:]+levels[:offset]
            engine_order=('tf32','fp16') if r%2 else ('fp16','tf32')
            order=[(n,w) for w in levels for n in engine_order]
            for pos,item in enumerate(order):positions.setdefault(item,[]).append(pos)
        self.assertTrue(all(sorted(v)==list(range(8)) for v in positions.values()))

if __name__=='__main__':unittest.main(verbosity=2)
