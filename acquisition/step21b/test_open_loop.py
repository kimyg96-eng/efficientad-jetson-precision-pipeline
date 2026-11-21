import unittest,time,threading
from open_loop_core import run_arrivals
class Tests(unittest.TestCase):
 def test_fast_order_and_count(self):
  def prep(row,i):time.sleep(.002 if i%2==0 else .0001);return i
  def consume(i):return {'finished_ns':time.monotonic_ns()}
  out,arr,m=run_arrivals([{'path':'x'}],50,.2,4,5,prep,consume)
  self.assertEqual(len(arr),10);self.assertEqual(len(out),10);self.assertLessEqual(m['max_inflight'],5)
  self.assertGreaterEqual(m['end_ns'],m['arrival_end_ns'])
 def test_overload_drops_and_denominators(self):
  def consume(i):time.sleep(.03);return {'finished_ns':time.monotonic_ns()}
  out,arr,m=run_arrivals([{'path':'x'}],200,.2,4,2,lambda row,i:i,consume)
  self.assertEqual(len(arr),40);self.assertLess(len(out),40);self.assertLessEqual(m['max_inflight'],2)
  self.assertEqual(len(out)+sum(r['status']!='accepted' for r in arr),40)
 def test_failure_stops(self):
  def prep(row,i):raise ValueError('fixture')
  with self.assertRaisesRegex(RuntimeError,'worker failed'):run_arrivals([{'path':'x'}],20,.2,2,5,prep,lambda x:x)
  self.assertFalse(any(t.name.startswith('arrival-prep') or t.name=='scheduled-arrivals' for t in threading.enumerate()))
if __name__=='__main__':unittest.main(verbosity=2)
