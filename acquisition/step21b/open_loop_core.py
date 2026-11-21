"""Independent scheduled arrivals, drop-new admission, ordered single consumer."""
import threading,time,math
from collections import deque

def run_arrivals(rows,rate,duration,workers,capacity,preprocess,consume):
 if not rows or rate<=0 or duration<=0 or workers<1 or capacity<1:raise ValueError('Invalid configuration')
 total=int(math.floor(rate*duration));period_ns=1e9/rate
 if total<1:raise ValueError('No scheduled arrivals')
 cv=threading.Condition();pending=deque();accepted_order=deque();ready={};arrivals=[];outputs=[];errors=[]
 state={'inflight':0,'max_inflight':0,'releases_done':False,'cancel':False,'alive':workers}
 start_ns=time.monotonic_ns()+100_000_000
 def fail(exc):
  with cv:errors.append(exc);state['cancel']=True;cv.notify_all()
 def release():
  try:
   for seq in range(total):
    scheduled=start_ns+round(seq*period_ns)
    # Cancellation-aware timed wait; never wait for admission capacity.
    with cv:
     while not state['cancel']:
      remain=(scheduled-time.monotonic_ns())/1e9
      if remain<=0:break
      cv.wait(timeout=remain)
     if state['cancel']:return
     actual=time.monotonic_ns();late=actual-scheduled
     r={'sequence':seq,'path':rows[seq%len(rows)]['path'],'scheduled_ns':scheduled,'release_ns':actual,'release_lateness_ms':late/1e6}
     if late>=period_ns:r['status']='drop_release_late'
     elif state['inflight']>=capacity:r['status']='drop_capacity'
     else:
      r['status']='accepted';state['inflight']+=1;state['max_inflight']=max(state['max_inflight'],state['inflight'])
      pending.append((r,rows[seq%len(rows)]));accepted_order.append(seq)
     arrivals.append(r);cv.notify_all()
   # Preserve the complete offered-arrival interval, even if the last image finished early.
   with cv:
    while not state['cancel']:
     remain=(start_ns+round(duration*1e9)-time.monotonic_ns())/1e9
     if remain<=0:break
     cv.wait(timeout=remain)
  except BaseException as exc:fail(exc)
  finally:
   with cv:state['releases_done']=True;cv.notify_all()
 def producer():
  try:
   while True:
    with cv:
     while not pending and not state['releases_done'] and not state['cancel']:cv.wait(.05)
     if state['cancel'] or not pending:return
     r,row=pending.popleft()
    item=preprocess(row,r['sequence'])
    with cv:ready[r['sequence']]=(r,item);cv.notify_all()
  except BaseException as exc:fail(exc)
  finally:
   with cv:state['alive']-=1;cv.notify_all()
 threads=[threading.Thread(target=producer,name=f'arrival-prep-{i}') for i in range(workers)]
 arrival_thread=threading.Thread(target=release,name='scheduled-arrivals');threads.append(arrival_thread)
 for t in threads:t.start()
 try:
  while True:
   with cv:
    while not errors and not state['cancel']:
     if accepted_order and accepted_order[0] in ready:break
     if state['releases_done'] and not accepted_order:break
     cv.wait(.05)
    if errors:raise RuntimeError('Open-loop worker failed') from errors[0]
    if state['releases_done'] and not accepted_order:break
    if state['cancel']:raise RuntimeError('Open-loop cancelled')
    seq=accepted_order.popleft();r,item=ready.pop(seq)
   result=consume(item);result.update({k:r[k] for k in ('sequence','path','scheduled_ns','release_ns','release_lateness_ms')})
   result['arrival_to_decision_ms']=(result['finished_ns']-r['scheduled_ns'])/1e6
   result['on_time_period']=int(result['finished_ns']<=r['scheduled_ns']+round(period_ns))
   result['on_time_100ms']=int(result['finished_ns']<=r['scheduled_ns']+100_000_000)
   outputs.append(result)
   with cv:state['inflight']-=1;cv.notify_all()
 finally:
  with cv:state['cancel']=True;cv.notify_all()
  for t in threads:t.join(timeout=30)
  if any(t.is_alive() for t in threads):raise RuntimeError('Open-loop worker failed to stop')
 end_ns=time.monotonic_ns()
 if errors:raise RuntimeError('Open-loop worker failed') from errors[0]
 accepted=[r['sequence'] for r in arrivals if r['status']=='accepted']
 if len(arrivals)!=total or [r['sequence'] for r in outputs]!=accepted or state['inflight']!=0:raise RuntimeError('Incomplete arrivals or results')
 return outputs,arrivals,{'start_ns':start_ns,'arrival_end_ns':start_ns+round(duration*1e9),'end_ns':end_ns,'offered':total,'max_inflight':state['max_inflight'],'capacity':capacity,'workers':workers}
