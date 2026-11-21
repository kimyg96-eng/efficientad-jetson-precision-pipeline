import unittest,tempfile,hashlib
from pathlib import Path
from recover import all_order,select_plan,prepare
from power_monitor import integrate
class TestRecovery(unittest.TestCase):
 def test_interruption_keeps_pairs(self):
  expected=all_order();available=set(expected[:493]);plan,retained=select_plan(available)
  self.assertEqual((len(plan),len(retained)),(20,492));self.assertEqual(set(plan)|set(retained),set(expected));self.assertFalse(set(plan)&set(retained))
  self.assertIn(('grid','default_tf32',8,10,7),plan);self.assertIn(('grid','automatic_fp16',8,10,7),plan)
  for c,e,p,b,r in plan:self.assertIn((c,'automatic_fp16' if e=='default_tf32' else 'default_tf32',p,b,r),plan)
 def test_complete_has_no_reacquisition(self):
  plan,keep=select_plan(set(all_order()));self.assertFalse(plan);self.assertEqual(len(keep),512)
 def test_bad_source_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);(p/'SHA256SUMS.txt').write_text('invalid')
   with self.assertRaises(ValueError):prepare(p,p/'out',False)
 def test_222ms_gap_still_rejected(self):
  rows=[]
  for t in (-50,0,50,272,322):
   row={'monotonic_ns':int(1e9+t*1e6)}
   for rail in ('VDD_GPU_SOC','VDD_CPU_CV','VIN_SYS_5V0'):row[rail+'_mV']=1000;row[rail+'_mA']=1000
   rows.append(row)
  with self.assertRaisesRegex(ValueError,'Power integration gap too large'):integrate(rows,1_000_000_000,1_300_000_000)
if __name__=='__main__':unittest.main()
