import csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path

class AuditTests(unittest.TestCase):
 def test_counts_deadlines_and_energy(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   (root/'protocol.json').write_text(json.dumps({'mode':'pilot','rates_img_s':[35,80],'rounds':1}))
   def write(path,rows):
    with path.open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   for c in ['bottle','carpet']:
    for e in ['default_tf32','automatic_fp16']:
     for rate in [35,80]:
      p=root/c/e/f'rate_{rate}'/'round_1';p.mkdir(parents=True)
      s={'category':c,'engine':e,'rate_img_s':rate,'round':1,'completed':2,'offered':3,'start_ns':1_000_000_000,'end_ns':2_000_000_000,'three_rail_energy_j':3.,'drop_capacity':1,'drop_release_late':0}
      (p/'summary.json').write_text(json.dumps(s))
      write(p/'images.csv',[{'sequence':i,'score':.5,'expected_score':.5,'threshold':.4,'decision':1,'arrival_to_decision_ms':v} for i,v in [(0,25),(2,125)]])
      write(p/'arrivals.csv',[{'sequence':i,'status':'accepted' if i!=1 else 'capacity'} for i in range(3)])
      rows=[]
      for ns in range(950_000_000,2_050_000_001,50_000_000):
       row={'monotonic_ns':ns}
       for rail in ['VDD_GPU_SOC','VDD_CPU_CV','VIN_SYS_5V0']:row.update({rail+'_mV':1000,rail+'_mA':1000})
       rows.append(row)
      write(p/'power.csv',rows)
   subprocess.run([sys.executable,str(Path(__file__).with_name('audit_arrivals.py')),str(root)],check=True,capture_output=True)
   with (root/'deadline_curves.csv').open() as f:curves=list(csv.DictReader(f))
   self.assertEqual(len(curves),8*41)
   self.assertAlmostEqual(float(next(x for x in curves if x['deadline_ms']=='50')['on_time_pct_of_offered']),100/3)
   self.assertTrue(all(x['largest_tested_zero_drop_rate'] is None for x in json.loads((root/'zero_drop_summary.json').read_text())))

if __name__=='__main__':unittest.main()
