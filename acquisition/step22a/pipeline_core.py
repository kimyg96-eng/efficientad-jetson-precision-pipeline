"""Bounded ordered CPU producers and a single caller-thread consumer."""
import threading
import time

def run_ordered(rows, workers, capacity, duration_s, preprocess, consume):
    if workers not in (0, 1, 2, 4, 6, 8) or capacity < 1 or not rows or duration_s <= 0:
        raise ValueError('Invalid pipeline configuration')
    start_ns = time.monotonic_ns()
    deadline_ns = start_ns + int(duration_s * 1e9)
    outputs = []
    if workers == 0:
        sequence = 0
        while True:
            if sequence % len(rows) == 0 and time.monotonic_ns() >= deadline_ns:
                break
            outputs.append(consume(preprocess(rows[sequence % len(rows)], sequence)))
            sequence += 1
        return outputs, {'start_ns': start_ns, 'end_ns': time.monotonic_ns(),
                         'max_inflight': 1, 'assigned': sequence, 'workers': 0}
    gate = threading.Condition()
    credits = threading.Semaphore(capacity)
    cancelled = threading.Event()
    ready = {}
    errors = []
    state = {'next': 0, 'closed': False, 'alive': workers, 'inflight': 0, 'max_inflight': 0}
    def producer():
        try:
            while not cancelled.is_set():
                if not credits.acquire(timeout=.05):
                    continue
                with gate:
                    if cancelled.is_set() or state['closed']:
                        credits.release()
                        return
                    seq = state['next']
                    if seq % len(rows) == 0 and time.monotonic_ns() >= deadline_ns:
                        state['closed'] = True
                        credits.release()
                        gate.notify_all()
                        return
                    state['next'] += 1
                    state['inflight'] += 1
                    state['max_inflight'] = max(state['max_inflight'], state['inflight'])
                item = preprocess(rows[seq % len(rows)], seq)
                with gate:
                    ready[seq] = item
                    gate.notify_all()
        except BaseException as exc:
            with gate:
                errors.append(exc)
                cancelled.set()
                gate.notify_all()
        finally:
            with gate:
                state['alive'] -= 1
                gate.notify_all()
    threads = [threading.Thread(target=producer, name=f'png-producer-{i}', daemon=True)
               for i in range(workers)]
    for thread in threads:
        thread.start()
    sequence = 0
    try:
        while True:
            with gate:
                while sequence not in ready and not errors:
                    if state['alive'] == 0:
                        if sequence != state['next']:
                            raise RuntimeError('Producer ended with missing input')
                        break
                    gate.wait(timeout=.05)
                if errors:
                    raise RuntimeError('Preprocessing worker failed') from errors[0]
                if sequence not in ready:
                    break
                item = ready.pop(sequence)
            try:
                outputs.append(consume(item))
            finally:
                with gate:
                    state['inflight'] -= 1
                credits.release()
            sequence += 1
    finally:
        cancelled.set()
        with gate:
            gate.notify_all()
        for thread in threads:
            thread.join(timeout=30)
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError('Preprocessing worker did not terminate')
    if state['next'] != len(outputs) or len(outputs) % len(rows):
        raise RuntimeError('Incomplete corpus or unconsumed assignments')
    return outputs, {'start_ns': start_ns, 'end_ns': time.monotonic_ns(),
                     'max_inflight': state['max_inflight'], 'assigned': state['next'],
                     'workers': workers}
