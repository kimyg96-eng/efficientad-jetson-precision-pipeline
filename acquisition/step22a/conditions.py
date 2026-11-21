"""Prespecified cells; total capacity includes preprocessing and consumer service."""
FULL_CELLS=((1,10),(2,10),(4,10),(6,10),(8,10),(4,1),(4,2),(4,5))
PILOT_CELLS=((4,1),(4,10),(6,10),(8,10))
def cells(mode):
    if mode not in ('pilot','full'):raise ValueError('Unknown mode')
    return list(PILOT_CELLS if mode=='pilot' else FULL_CELLS)
def ordered_arms(mode,round_id):
    if round_id<1:raise ValueError('Invalid round')
    vs=cells(mode);offset=(round_id-1)%len(vs);vs=vs[offset:]+vs[:offset]
    names=('default_tf32','automatic_fp16') if round_id%2 else ('automatic_fp16','default_tf32')
    return [(name,p,c) for p,c in vs for name in names]
